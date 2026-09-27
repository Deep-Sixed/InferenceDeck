from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import telemetry, telemetry_history
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler
from inferencedeck.telemetry_history import TelemetryHistory, bucket, flatten

T0 = 1_790_000_040.0  # on a minute boundary


def _snap(cpu: float, vram_gb: float = 1.0) -> dict:
    return {
        "system": {"cpu_percent": cpu, "memory": {"used_bytes": 4, "total_bytes": 16}},
        "gpus": [{"index": 0, "name": "RTX 3090", "utilization_percent": cpu, "vram_used_bytes": vram_gb * 1024**3, "vram_total_bytes": 24 * 1024**3, "power_watts": None}],
        "servers": [
            {"profile": "qwen", "running": True, "gpu_memory_bytes": 7, "rss_bytes": 9},
            {"profile": "old", "running": False, "gpu_memory_bytes": 99},
        ],
    }


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FlattenAndBucketTests(unittest.TestCase):
    def test_flatten_keeps_numbers_and_describes_series(self) -> None:
        values, meta = flatten(_snap(50))
        self.assertEqual(values["cpu_percent"], 50)
        self.assertEqual(values["gpu0.utilization_percent"], 50)
        self.assertEqual(values["server:qwen.gpu_memory_bytes"], 7)
        self.assertNotIn("gpu0.power_watts", values)  # missing, not zero
        self.assertNotIn("server:old.gpu_memory_bytes", values)  # stopped servers hold nothing
        self.assertEqual(meta["labels"]["gpu0"], "RTX 3090")
        self.assertEqual(meta["limits"]["gpu0.vram_used_bytes"], 24 * 1024**3)

    def test_bucket_averages_and_leaves_gaps(self) -> None:
        rows = [(T0 + 1, {"a": 10.0}), (T0 + 3, {"a": 20.0}), (T0 + 21, {"a": 5.0, "b": 1.0})]
        slots, series = bucket(rows, T0, T0 + 25, 10)
        self.assertEqual(slots, [int(T0), int(T0) + 10, int(T0) + 20])
        self.assertEqual(series["a"], [15.0, None, 5.0])
        self.assertEqual(series["b"], [None, None, 1.0])


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.clock = Clock(T0)

    def _history(self, **kw) -> TelemetryHistory:
        return TelemetryHistory(directory=self.dir, interval=5, clock=self.clock, collect=lambda: _snap(1), **kw)

    def test_raw_range_uses_sample_interval(self) -> None:
        history = self._history()
        for i in range(12):
            self.clock.t = T0 + i * 5
            history.record(_snap(i))
        result = history.query("15m")
        self.assertEqual(result["step_seconds"], 5)
        self.assertEqual(result["series"]["cpu_percent"][-12:], [float(i) for i in range(12)])
        self.assertEqual(result["labels"], {"gpu0": "RTX 3090"})

    def test_minutes_are_averaged_persisted_and_reloaded(self) -> None:
        history = self._history()
        for i, cpu in enumerate([10, 20, 30]):
            history.record(_snap(cpu), t=T0 + i * 20)
        history.record(_snap(90), t=T0 + 60)  # opens the next minute, closing the first
        lines = [json.loads(line) for f in self.dir.glob("*.jsonl") for line in f.read_text().splitlines()]
        self.assertEqual(lines[0]["k"], "m")
        self.assertEqual(lines[0]["v"]["cpu_percent"], 20.0)

        self.clock.t = T0 + 120
        reloaded = self._history()
        reloaded.load()
        series = reloaded.query("6h")["series"]["cpu_percent"]
        self.assertEqual([v for v in series if v is not None], [20.0])

    def test_open_minute_is_included_in_minute_ranges(self) -> None:
        history = self._history()
        history.record(_snap(40), t=T0 + 5)
        self.clock.t = T0 + 30
        self.assertEqual([v for v in history.query("6h")["series"]["cpu_percent"] if v is not None], [40.0])

    def test_events_are_persisted_and_returned_in_range(self) -> None:
        history = self._history()
        self.clock.t = T0 + 10
        history.record_event({"type": "server.ready", "profile": "qwen", "startup_seconds": 7.4})
        reloaded = self._history()
        reloaded.load()
        events = reloaded.query("1h")["events"]
        self.assertEqual((events[0]["type"], events[0]["t"]), ("server.ready", T0 + 10))

    def test_load_drops_files_past_retention_and_torn_lines(self) -> None:
        old = self.dir / "2000-01-01.jsonl"
        old.write_text('{"k":"m","t":1,"v":{"cpu_percent":1}}\n')
        history = self._history(retention_days=1)
        history._append(T0, {"k": "m", "t": T0, "v": {"cpu_percent": 5}})
        with history._file_for(T0).open("a") as fh:
            fh.write('{"k":"m","t":')  # crash mid-write
        self.clock.t = T0 + 60
        history.load()
        self.assertFalse(old.exists())
        self.assertEqual([v for v in history.query("6h")["series"]["cpu_percent"] if v is not None], [5.0])

    def test_unknown_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._history().query("1y")

    def test_failing_collector_does_not_stop_sampling(self) -> None:
        history = TelemetryHistory(directory=self.dir, collect=mock.Mock(side_effect=RuntimeError("boom")))
        history.sample_once()  # no exception

    def test_start_subscribes_to_lifecycle_events(self) -> None:
        history = self._history()
        history.interval = 3600  # one sample, then idle
        history.start()
        self.addCleanup(history.stop)
        telemetry.emit("server.released", {"id": "qwen-1", "mode": "qwen"})
        self.assertEqual(history.query("1h")["events"][-1]["type"], "server.released")
        history.stop()
        telemetry.emit("server.restored", {"id": "qwen-1", "mode": "qwen"})
        self.assertEqual(history.query("1h")["events"][-1]["type"], "server.released")


class HistoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("TestHandler", (ControlRequestHandler,), {})
        handler.control_plane = ControlPlane()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        self.base = f"http://127.0.0.1:{server.server_port}"

    def test_history_disabled_when_no_sampler(self) -> None:
        with mock.patch.object(telemetry_history, "_history", None):
            with urllib.request.urlopen(self.base + "/api/telemetry/history?range=6h", timeout=2) as response:
                body = json.loads(response.read())
        self.assertEqual((body["enabled"], body["range"]), (False, "6h"))

    def test_history_served_from_sampler_and_bad_range_is_400(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        history = TelemetryHistory(directory=Path(tmp.name))
        history.record(_snap(12))
        with mock.patch.object(telemetry_history, "_history", history):
            with urllib.request.urlopen(self.base + "/api/telemetry/history", timeout=2) as response:
                body = json.loads(response.read())
            with self.assertRaises(urllib.error.HTTPError) as bad:
                urllib.request.urlopen(self.base + "/api/telemetry/history?range=1y", timeout=2)
        self.assertTrue(body["enabled"])
        self.assertIn(12.0, body["series"]["cpu_percent"])
        self.assertEqual(bad.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
