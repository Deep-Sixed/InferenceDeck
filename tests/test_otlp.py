from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from inferencedeck import otlp, telemetry

SNAP = {
    "timestamp": "2026-09-27T10:00:00+00:00",
    "control_uptime_seconds": 100.0,
    "system": {"cpu_percent": 34.7},
    "gpus": [{"index": 0, "name": "RTX 3090", "utilization_percent": 87.0, "vram_used_bytes": 19863199744}],
    "servers": [{"profile": "qwen", "runtime": "llama.cpp", "running": True, "uptime_seconds": 60.0, "tokens_generated_total": 98231}],
    "lifecycle": {"counters": [{"event": "server.started", "profile": "qwen", "runtime": "llama.cpp", "count": 2}]},
}
T_SNAP = 1790503200.0  # 2026-09-27T10:00:00Z


def _find(payload: dict, name: str) -> dict:
    metrics = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    return next(m for m in metrics if m["name"] == name)


def _attrs(items: list[dict]) -> dict:
    return {a["key"]: next(iter(a["value"].values())) for a in items}


class PayloadTests(unittest.TestCase):
    def test_gauges_and_cumulative_sums_with_start_times(self) -> None:
        payload = otlp.metrics_payload(telemetry.metric_samples(SNAP), otlp._attributes({"service.name": "inferencedeck"}), T_SNAP + 1)
        self.assertEqual(_attrs(payload["resourceMetrics"][0]["resource"]["attributes"]), {"service.name": "inferencedeck"})

        util = _find(payload, "inferencedeck_gpu_utilization_percent")
        self.assertEqual(util["unit"], "%")
        point = util["gauge"]["dataPoints"][0]
        self.assertEqual((point["asDouble"], _attrs(point["attributes"])), (87.0, {"gpu": "0", "name": "RTX 3090"}))
        self.assertEqual(point["timeUnixNano"], str(int((T_SNAP + 1) * 1e9)))

        vram = _find(payload, "inferencedeck_gpu_vram_used_bytes")
        self.assertEqual(vram["unit"], "By")

        tokens = _find(payload, "inferencedeck_server_generated_tokens_total")
        self.assertEqual((tokens["sum"]["aggregationTemporality"], tokens["sum"]["isMonotonic"]), (2, True))
        self.assertEqual(tokens["sum"]["dataPoints"][0]["startTimeUnixNano"], str(int((T_SNAP - 60) * 1e9)))

        events = _find(payload, "inferencedeck_lifecycle_events_total")
        self.assertEqual(events["sum"]["dataPoints"][0]["startTimeUnixNano"], str(int((T_SNAP - 100) * 1e9)))

    def test_log_record_marks_failures_as_warnings(self) -> None:
        ok = otlp.log_record({"type": "server.ready", "timestamp": "2026-09-27T10:00:00+00:00", "profile": "qwen", "pid": 42, "startup_seconds": 7.4})
        self.assertEqual((ok["severityText"], ok["body"]["stringValue"]), ("INFO", "server.ready qwen"))
        self.assertEqual(_attrs(ok["attributes"]), {"event.name": "server.ready", "inferencedeck.profile": "qwen", "process.pid": "42", "inferencedeck.startup_seconds": 7.4})
        failed = otlp.log_record({"type": "server.start_failed", "profile": "qwen", "reason": "startup_timeout"})
        self.assertEqual((failed["severityNumber"], failed["body"]["stringValue"]), (13, "server.start_failed qwen startup_timeout"))

    def test_spans_for_startup_benchmark_and_failure(self) -> None:
        ready = otlp.span_for({"type": "server.ready", "timestamp": "2026-09-27T10:00:00+00:00", "profile": "qwen", "startup_seconds": 7.5})
        self.assertEqual(ready["name"], "inferencedeck.server.startup")
        self.assertEqual(int(ready["endTimeUnixNano"]) - int(ready["startTimeUnixNano"]), 7_500_000_000)
        self.assertEqual((len(ready["traceId"]), len(ready["spanId"]), ready["status"]), (32, 16, {"code": 1}))

        bench = otlp.span_for({"type": "benchmark.completed", "elapsed_seconds": 4.0, "tokens_per_second": 32.4})
        self.assertEqual(bench["name"], "inferencedeck.benchmark")

        failed = otlp.span_for({"type": "server.start_failed", "waited_seconds": 45.0, "reason": "startup_timeout"})
        self.assertEqual(failed["status"], {"code": 2, "message": "startup_timeout"})

        self.assertIsNone(otlp.span_for({"type": "server.stopped"}))
        self.assertIsNone(otlp.span_for({"type": "server.start_failed", "reason": "exec failed"}))  # no duration

    def test_attribute_types(self) -> None:
        self.assertEqual(otlp._attributes({"a": True, "b": 3, "c": 1.5, "d": "x", "e": None}), [
            {"key": "a", "value": {"boolValue": True}},
            {"key": "b", "value": {"intValue": "3"}},
            {"key": "c", "value": {"doubleValue": 1.5}},
            {"key": "d", "value": {"stringValue": "x"}},
        ])


class ConfigTests(unittest.TestCase):
    def test_key_value_parsing(self) -> None:
        self.assertEqual(otlp.parse_key_values("Authorization=Bearer%20abc, x-team = ml ,bad"), {"Authorization": "Bearer abc", "x-team": "ml"})

    def test_endpoint_from_config_or_environment(self) -> None:
        with mock.patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318/"}):
            self.assertEqual(otlp.resolve_endpoint(), "http://collector:4318")
            self.assertEqual(otlp.resolve_endpoint("https://other:4318"), "https://other:4318")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(otlp.resolve_endpoint(), "")
            with self.assertRaises(ValueError):
                otlp.resolve_endpoint("collector:4317")

    def test_resource_honours_otel_variables(self) -> None:
        env = {"OTEL_SERVICE_NAME": "thanatos-deck", "OTEL_RESOURCE_ATTRIBUTES": "service.name=ignored,deployment.environment=home"}
        with mock.patch.dict(os.environ, env, clear=True):
            resource = otlp.default_resource()
        self.assertEqual((resource["service.name"], resource["deployment.environment"]), ("thanatos-deck", "home"))

    def test_no_endpoint_means_no_exporter(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(otlp, "_exporter", None):
            self.assertIsNone(otlp.start_exporter(""))
            self.assertEqual(otlp.status(), {"enabled": False})


class _Collector(BaseHTTPRequestHandler):
    received: list = []
    fail = False

    def log_message(self, *args) -> None:
        return

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).received.append((self.path, dict(self.headers), json.loads(body)))
        self.send_response(503 if type(self).fail else 200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = type("C", (_Collector,), {"received": [], "fail": False})
        server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        self.exporter = otlp.OtlpExporter(
            f"http://127.0.0.1:{server.server_port}",
            headers={"Authorization": "Bearer abc"},
            resource={"service.name": "inferencedeck"},
            collect=lambda: SNAP,
        )

    def test_exports_all_three_signals_with_headers(self) -> None:
        self.exporter.record_event({"type": "server.ready", "profile": "qwen", "startup_seconds": 7.4})
        self.exporter.record_event({"type": "server.stopped", "profile": "qwen"})
        self.assertTrue(self.exporter.export_once())
        paths = [path for path, _, _ in self.handler.received]
        self.assertEqual(paths, ["/v1/metrics", "/v1/logs", "/v1/traces"])
        _, headers, logs = self.handler.received[1]
        self.assertEqual((headers["Authorization"], headers["Content-Type"]), ("Bearer abc", "application/json"))
        self.assertEqual(len(logs["resourceLogs"][0]["scopeLogs"][0]["logRecords"]), 2)
        spans = self.handler.received[2][2]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual([s["name"] for s in spans], ["inferencedeck.server.startup"])
        status = self.exporter.status()
        self.assertEqual((status["exported"]["logs"], status["exported"]["spans"], status["last_error"]), (2, 1, None))

    def test_failed_export_keeps_events_for_next_round(self) -> None:
        self.exporter.record_event({"type": "server.stopped", "profile": "qwen"})
        self.handler.fail = True
        self.assertFalse(self.exporter.export_once())
        self.assertIn("503", self.exporter.status()["last_error"])
        self.assertEqual(self.exporter.status()["queued"], 1)
        self.handler.fail = False
        self.handler.received.clear()
        self.assertTrue(self.exporter.export_once())
        self.assertEqual([p for p, _, _ in self.handler.received], ["/v1/metrics", "/v1/logs"])
        self.assertEqual(self.exporter.status()["queued"], 0)

    def test_queue_is_bounded(self) -> None:
        with mock.patch.object(otlp, "MAX_QUEUED_RECORDS", 3):
            for i in range(5):
                self.exporter.record_event({"type": "server.stopped", "profile": f"p{i}"})
        self.assertEqual((self.exporter.status()["queued"], self.exporter.status()["dropped"]), (3, 2))

    def test_start_listens_for_lifecycle_events(self) -> None:
        self.exporter.interval = 3600
        self.exporter.start()
        telemetry.emit("server.released", {"id": "qwen-1", "mode": "qwen"})
        self.assertEqual(self.exporter.status()["queued"], 1)
        self.exporter.stop()  # flushes
        self.assertIn("/v1/logs", [p for p, _, _ in self.handler.received])
        telemetry.emit("server.restored", {"id": "qwen-1", "mode": "qwen"})
        self.assertEqual(self.exporter.status()["queued"], 0)


if __name__ == "__main__":
    unittest.main()
