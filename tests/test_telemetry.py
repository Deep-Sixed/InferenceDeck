from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import server_manager, telemetry
from inferencedeck.auth import AuthState
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler


def _done(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        telemetry.REGISTRY.reset()
        self.addCleanup(telemetry.REGISTRY.reset)

    def test_emit_counts_by_event_profile_and_runtime(self) -> None:
        server = {"id": "qwen-1", "mode": "qwen", "runtime": "llama.cpp", "pid": 1}
        telemetry.emit("server.started", server)
        telemetry.emit("server.started", server)
        telemetry.emit("server.ready", server, startup_seconds=7.4)
        counters = {(c["event"], c["profile"]): c["count"] for c in telemetry.REGISTRY.counters()}
        self.assertEqual(counters, {("server.ready", "qwen"): 1, ("server.started", "qwen"): 2})
        last = telemetry.REGISTRY.recent_events()[-1]
        self.assertEqual((last["type"], last["server_id"], last["startup_seconds"]), ("server.ready", "qwen-1", 7.4))

    def test_recent_events_are_bounded(self) -> None:
        registry = telemetry.TelemetryRegistry(max_events=3)
        for i in range(5):
            registry.emit("server.started", profile=f"p{i}")
        self.assertEqual([e["profile"] for e in registry.recent_events()], ["p2", "p3", "p4"])

    def test_emit_never_raises(self) -> None:
        with mock.patch.object(telemetry.REGISTRY, "emit", side_effect=RuntimeError("broken")):
            telemetry.emit("server.started", {"id": "x"})


class CollectorTests(unittest.TestCase):
    def test_nvidia_samples_parse_and_tolerate_not_available(self) -> None:
        outputs = [
            _done("0, NVIDIA GeForce RTX 3090, 87, 18943, 24576, 71, 292.40, 1785\n1, Tesla P4, [N/A], 10, 7680, 40, [N/A], N/A\n"),
            _done("4242, 18000\n4242, 500\n"),
        ]
        with mock.patch.object(telemetry.shutil, "which", return_value="nvidia-smi"), mock.patch.object(
            telemetry.hardware, "_run", side_effect=outputs
        ):
            gpus, per_pid = telemetry.nvidia_gpu_samples()
        self.assertEqual(gpus[0]["utilization_percent"], 87.0)
        self.assertEqual(gpus[0]["vram_used_bytes"], 18943 * telemetry.MIB)
        self.assertEqual((gpus[0]["power_watts"], gpus[0]["sm_clock_mhz"]), (292.4, 1785))
        self.assertIsNone(gpus[1]["utilization_percent"])
        self.assertIsNone(gpus[1]["power_watts"])
        self.assertEqual(per_pid, {4242: 18500 * telemetry.MIB})

    def test_no_nvidia_smi_means_no_gpu_samples(self) -> None:
        with mock.patch.object(telemetry.shutil, "which", return_value=None):
            self.assertEqual(telemetry.nvidia_gpu_samples(), ([], {}))

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "needs /proc")
    def test_process_stats_for_this_process(self) -> None:
        stats = telemetry.process_stats(os.getpid())
        self.assertGreater(stats["rss_bytes"], 0)
        self.assertGreaterEqual(stats["cpu_seconds"], 0)

    def test_snapshot_survives_failing_collectors(self) -> None:
        with mock.patch.object(telemetry, "nvidia_gpu_samples", side_effect=OSError("no driver")), mock.patch.object(
            telemetry, "system_memory", side_effect=OSError("no /proc")
        ):
            snap = telemetry.collect_snapshot(list_servers=mock.Mock(side_effect=RuntimeError("state unreadable")))
        self.assertEqual((snap["gpus"], snap["servers"], snap["system"]["memory"]), ([], [], {}))

    def test_server_entry_reports_uptime_startup_and_gpu_memory(self) -> None:
        servers = [
            {
                "id": "qwen-4242",
                "mode": "qwen",
                "runtime": "llama.cpp",
                "pid": 4242,
                "status": "running",
                "running": True,
                "ctx_size": 32768,
                "started_at": "2026-09-27T10:00:00+00:00",
                "ready_at": "2026-09-27T10:00:07.432000+00:00",
            },
            {"id": "llama-1", "mode": "llama", "pid": None, "status": "parked", "running": False},
        ]
        with mock.patch.object(telemetry, "nvidia_gpu_samples", return_value=([], {4242: 1024})), mock.patch.object(
            telemetry, "process_stats", return_value={"rss_bytes": 2048}
        ):
            snap = telemetry.collect_snapshot(list_servers=lambda: servers)
        qwen, parked = snap["servers"]
        self.assertEqual((qwen["startup_seconds"], qwen["context_size"]), (7.432, 32768))
        self.assertEqual((qwen["gpu_memory_bytes"], qwen["rss_bytes"]), (1024, 2048))
        self.assertGreater(qwen["uptime_seconds"], 0)
        self.assertEqual((parked["running"], parked["pid"], parked["uptime_seconds"]), (False, None, None))


class PrometheusTests(unittest.TestCase):
    def _snap(self, servers: list[dict]) -> dict:
        return {
            "control_uptime_seconds": 12.5,
            "system": {"cpu_percent": 34.7, "load_average": [1.0, 0.5, 0.25], "memory": {"total_bytes": 100, "available_bytes": 40}},
            "gpus": [{"index": 0, "name": 'RTX "3090"', "utilization_percent": 87.0, "vram_used_bytes": 5, "power_watts": None}],
            "servers": servers,
            "lifecycle": {"counters": [{"event": "server.started", "profile": "qwen", "runtime": "llama.cpp", "count": 3}]},
        }

    def test_renders_valid_families_and_escapes_labels(self) -> None:
        text = telemetry.render_prometheus(self._snap([]))
        self.assertIn("# TYPE inferencedeck_gpu_utilization_percent gauge", text)
        self.assertIn('inferencedeck_gpu_utilization_percent{gpu="0",name="RTX \\"3090\\""} 87', text)
        self.assertIn('inferencedeck_load_average{window="5m"} 0.5', text)
        self.assertIn('inferencedeck_lifecycle_events_total{event="server.started",profile="qwen",runtime="llama.cpp"} 3', text)
        # Missing readings are left out rather than exported as zero.
        self.assertNotIn("inferencedeck_gpu_power_watts", text)
        self.assertTrue(text.endswith("\n"))

    def test_one_series_per_profile_preferring_the_live_server(self) -> None:
        servers = [
            {"profile": "qwen", "runtime": "llama.cpp", "running": False, "status": "parked"},
            {"profile": "qwen", "runtime": "llama.cpp", "running": True, "uptime_seconds": 60},
        ]
        text = telemetry.render_prometheus(self._snap(servers))
        up = [line for line in text.splitlines() if line.startswith("inferencedeck_server_up{")]
        self.assertEqual(up, ['inferencedeck_server_up{profile="qwen",runtime="llama.cpp"} 1'])


class LifecycleEventTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(server_manager, "cache_dir", return_value=Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        telemetry.REGISTRY.reset()
        self.addCleanup(telemetry.REGISTRY.reset)

    def _events(self) -> list[str]:
        return [e["type"] for e in telemetry.REGISTRY.recent_events()]

    def test_parked_server_restore_emits_restored(self) -> None:
        server_manager._upsert_server(
            {"id": "qwen-1", "mode": "qwen", "pid": None, "status": server_manager.PARKED, "restart": {"mode": "qwen"}}
        )
        started = {"success": True, "server": {"id": "qwen-2", "running": True}}
        with mock.patch.object(server_manager, "start_profile", return_value=started):
            server_manager.restore_server("qwen-1")
        self.assertEqual(self._events(), ["server.restored"])
        self.assertEqual(telemetry.REGISTRY.recent_events()[-1]["new_server_id"], "qwen-2")

    def test_stop_emits_stopped(self) -> None:
        import sys

        proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
        self.addCleanup(lambda: (proc.poll() is None and proc.kill(), proc.wait(timeout=5)))
        server_manager._upsert_server({"id": f"qwen-{proc.pid}", "mode": "qwen", "pid": proc.pid, "status": "running"})
        result = server_manager.stop_server(mode="qwen")
        self.assertTrue(result["success"])
        self.assertEqual(self._events(), ["server.stopped"])


class TelemetryApiTests(unittest.TestCase):
    def _serve(self, auth: AuthState | None = None) -> str:
        handler = type("TestHandler", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.telemetry.return_value = {"version": 1, "gpus": []}
        handler.control_plane.metrics.return_value = "inferencedeck_control_uptime_seconds 1\n"
        handler.auth_state = auth or AuthState(token="")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        return f"http://127.0.0.1:{server.server_port}"

    def test_telemetry_json_and_metrics_text(self) -> None:
        base = self._serve()
        with urllib.request.urlopen(base + "/api/telemetry", timeout=2) as response:
            self.assertEqual(response.headers["Content-Type"], "application/json; charset=utf-8")
        with urllib.request.urlopen(base + "/metrics", timeout=2) as response:
            self.assertTrue(response.headers["Content-Type"].startswith("text/plain; version=0.0.4"))
            self.assertEqual(response.read().decode(), "inferencedeck_control_uptime_seconds 1\n")

    def test_metrics_require_auth_and_accept_a_bearer_token(self) -> None:
        base = self._serve(AuthState(token="s3cret"))
        with self.assertRaises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(base + "/metrics", timeout=2)
        self.assertEqual(denied.exception.code, 401)
        request = urllib.request.Request(base + "/metrics", headers={"Authorization": "Bearer s3cret"})
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 200)
        wrong = urllib.request.Request(base + "/metrics", headers={"Authorization": "Bearer nope"})
        with self.assertRaises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(wrong, timeout=2)
        self.assertEqual(rejected.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
