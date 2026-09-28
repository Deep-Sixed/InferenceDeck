from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from inferencedeck import telemetry
from inferencedeck.llama_args import build_llama_server_args
from inferencedeck.telemetry_history import flatten

LLAMA_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1842
llamacpp:prompt_seconds_total 0.684
llamacpp:tokens_predicted_total 128
llamacpp:tokens_predicted_seconds_total 3.948
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:predicted_tokens_seconds 32.4
"""


class _MetricsHandler(BaseHTTPRequestHandler):
    body = LLAMA_METRICS
    status = 200

    def log_message(self, *args) -> None:
        return

    def do_GET(self) -> None:
        data = self.body.encode()
        self.send_response(self.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ParseTests(unittest.TestCase):
    def test_parses_samples_sums_labels_and_skips_comments_and_nan(self) -> None:
        text = '# HELP x y\nplain 3\nlabelled{slot="0"} 2\nlabelled{slot="1",note="a}b"} 5 1700000000\nbad NaN\nbroken\n'
        self.assertEqual(telemetry.parse_prometheus_text(text), {"plain": 3.0, "labelled": 7.0})


class FetchTests(unittest.TestCase):
    def _serve(self, body: str = LLAMA_METRICS, status: int = 200) -> dict:
        handler = type("H", (_MetricsHandler,), {"body": body, "status": status})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        return {"id": "qwen-1", "host": "127.0.0.1", "port": server.server_port}

    def test_llama_server_counters_are_mapped(self) -> None:
        counters = telemetry.fetch_server_metrics(self._serve())
        self.assertEqual(counters["tokens_generated_total"], 128)
        self.assertEqual(counters["generation_seconds_total"], 3.948)
        self.assertEqual(counters["requests_active"], 1)
        self.assertEqual(counters["kv_cache_usage_ratio"], 0.25)
        self.assertNotIn("predicted_tokens_seconds", counters)  # bucket gauge, reset by every scrape

    def test_vllm_style_names_are_mapped(self) -> None:
        body = "vllm:generation_tokens_total 50\nvllm:num_requests_running 2\nvllm:gpu_cache_usage_perc 0.5\n"
        counters = telemetry.fetch_server_metrics(self._serve(body))
        self.assertEqual(counters, {"tokens_generated_total": 50, "requests_active": 2, "kv_cache_usage_ratio": 0.5})

    def test_missing_endpoint_or_dead_server_gives_none(self) -> None:
        self.assertIsNone(telemetry.fetch_server_metrics(self._serve("not supported", status=501)))
        self.assertIsNone(telemetry.fetch_server_metrics(self._serve("unrelated_metric 1\n")))
        self.assertIsNone(telemetry.fetch_server_metrics({"host": "127.0.0.1", "port": 9}, timeout=0.5))


class RateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rates = telemetry._RateTracker()

    def test_first_reading_has_no_rates(self) -> None:
        out = self.rates.rates("s", 0, {"tokens_generated_total": 0, "generation_seconds_total": 0})
        self.assertEqual(set(out.values()), {None})

    def test_speed_uses_busy_time_and_throughput_uses_wall_time(self) -> None:
        self.rates.rates("s", 0, {"tokens_generated_total": 0, "generation_seconds_total": 0, "prompt_tokens_total": 0, "prompt_seconds_total": 0})
        out = self.rates.rates("s", 10, {"tokens_generated_total": 128, "generation_seconds_total": 4, "prompt_tokens_total": 1000, "prompt_seconds_total": 0.5})
        self.assertEqual(out, {"tokens_per_second": 32.0, "prompt_tokens_per_second": 2000.0, "throughput_tokens_per_second": 12.8})

    def test_idle_server_has_no_speed_but_zero_throughput(self) -> None:
        counters = {"tokens_generated_total": 128, "generation_seconds_total": 4}
        self.rates.rates("s", 0, counters)
        out = self.rates.rates("s", 5, dict(counters))
        self.assertIsNone(out["tokens_per_second"])
        self.assertEqual(out["throughput_tokens_per_second"], 0.0)

    def test_counter_reset_is_not_a_negative_rate(self) -> None:
        self.rates.rates("s", 0, {"tokens_generated_total": 500, "generation_seconds_total": 20})
        out = self.rates.rates("s", 5, {"tokens_generated_total": 10, "generation_seconds_total": 1})
        self.assertEqual(set(out.values()), {None})

    def test_without_a_time_counter_speed_falls_back_to_wall_clock(self) -> None:
        self.rates.rates("s", 0, {"tokens_generated_total": 0})
        self.assertEqual(self.rates.rates("s", 4, {"tokens_generated_total": 100})["tokens_per_second"], 25.0)

    def test_forgotten_servers_start_over(self) -> None:
        self.rates.rates("s", 0, {"tokens_generated_total": 0})
        self.rates.forget_except(set())
        self.assertIsNone(self.rates.rates("s", 4, {"tokens_generated_total": 100})["throughput_tokens_per_second"])


class SnapshotTests(unittest.TestCase):
    def _server(self, **kw) -> dict:
        return {"id": "qwen-1", "mode": "qwen", "pid": 1, "status": "running", "running": True, "host": "127.0.0.1", "port": 8080, **kw}

    def test_live_server_gets_inference_stats(self) -> None:
        stats = {"metrics_available": True, "requests_active": 1, "tokens_per_second": 31.8}
        with mock.patch.object(telemetry, "inference_stats", return_value=stats), mock.patch.object(telemetry, "process_stats", return_value={}):
            entry = telemetry._server_entry(self._server(), 0, {})
        self.assertEqual((entry["requests_active"], entry["tokens_per_second"]), (1, 31.8))

    def test_paused_server_is_not_asked(self) -> None:
        with mock.patch.object(telemetry, "inference_stats") as stats, mock.patch.object(telemetry, "process_stats", return_value={}):
            telemetry._server_entry(self._server(suspended=True), 0, {})
        stats.assert_not_called()

    def test_inference_stats_combines_counters_and_rates(self) -> None:
        with mock.patch.object(telemetry, "fetch_server_metrics", return_value={"requests_active": 2.0, "kv_cache_usage_ratio": 0.125, "tokens_generated_total": 9.0}):
            stats = telemetry.inference_stats({"id": "fresh-server"}, 0)
        self.assertEqual((stats["requests_active"], stats["kv_cache_usage_percent"], stats["tokens_generated_total"]), (2, 12.5, 9))
        with mock.patch.object(telemetry, "fetch_server_metrics", return_value=None):
            self.assertEqual(telemetry.inference_stats({"id": "x"}, 0), {"metrics_available": False})

    def test_latest_benchmark_per_profile(self) -> None:
        results = [
            {"mode": "qwen", "tokens_per_second": 30, "created_at": "2026-09-01T00:00:00+00:00"},
            {"mode": "qwen", "tokens_per_second": 32, "created_at": "2026-09-02T00:00:00+00:00"},
            {"mode": "llama", "tokens_per_second": 78, "created_at": "2026-09-01T00:00:00+00:00"},
        ]
        with mock.patch("inferencedeck.benchmark.load_benchmark_results", return_value=results):
            latest = telemetry.latest_benchmarks()
        self.assertEqual([(b["profile"], b["tokens_per_second"]) for b in latest], [("llama", 78), ("qwen", 32)])


class ExportTests(unittest.TestCase):
    def test_prometheus_and_history_carry_request_metrics(self) -> None:
        server = {"profile": "qwen", "runtime": "llama.cpp", "running": True, "requests_active": 1, "tokens_generated_total": 98231, "tokens_per_second": 31.8, "kv_cache_usage_percent": 25.0}
        snap = {"servers": [server], "benchmarks": [{"profile": "qwen", "tokens_per_second": 32.7}]}
        text = telemetry.render_prometheus(snap)
        self.assertIn('inferencedeck_server_generation_tokens_per_second{profile="qwen",runtime="llama.cpp"} 31.8', text)
        self.assertIn('inferencedeck_server_generated_tokens_total{profile="qwen",runtime="llama.cpp"} 98231', text)
        self.assertIn('inferencedeck_server_kv_cache_usage_ratio{profile="qwen",runtime="llama.cpp"} 0.25', text)
        self.assertIn('inferencedeck_benchmark_tokens_per_second{profile="qwen"} 32.7', text)
        values, _ = flatten(snap)
        self.assertEqual((values["server:qwen.tokens_per_second"], values["server:qwen.requests_active"]), (31.8, 1.0))

    def test_llama_server_is_launched_with_metrics(self) -> None:
        self.assertIn("--metrics", build_llama_server_args("/bin/llama-server", "/m.gguf", {}).argv)
        self.assertNotIn("--metrics", build_llama_server_args("/bin/llama-server", "/m.gguf", {"metrics": False}).argv)


if __name__ == "__main__":
    unittest.main()
