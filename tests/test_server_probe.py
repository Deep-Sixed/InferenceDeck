from __future__ import annotations

import json
import os
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inferencedeck import server_manager
from inferencedeck.benchmark import response_metrics


class _FakeServer:
    """Minimal HTTP server whose routes map a path to a list of (status, body) replies."""

    def __init__(self, routes: dict[str, list[tuple[int, object]]]) -> None:
        self.routes = routes
        self.hits: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_GET(self) -> None:
                outer.hits.append(self.path)
                replies = outer.routes.get(self.path) or [(404, {"error": "not found"})]
                status, body = replies.pop(0) if len(replies) > 1 else replies[0]
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class WaitUntilReadyTests(unittest.TestCase):
    def _serve(self, routes: dict[str, list[tuple[int, object]]]) -> _FakeServer:
        server = _FakeServer(routes)
        self.addCleanup(server.close)
        return server

    def test_waits_through_loading_503(self) -> None:
        server = self._serve({"/health": [(503, {"error": "Loading model"}), (200, {"status": "ok"})]})
        self.assertTrue(server_manager.wait_until_ready("127.0.0.1", server.port, os.getpid(), 15))
        self.assertEqual(server.hits, ["/health", "/health"])

    def test_falls_back_to_models_when_health_missing(self) -> None:
        server = self._serve({"/v1/models": [(200, {"data": []})]})
        self.assertTrue(server_manager.wait_until_ready("0.0.0.0", server.port, os.getpid(), 15))
        self.assertEqual(server.hits, ["/health", "/v1/models"])

    def test_dead_process_is_not_ready(self) -> None:
        server = self._serve({"/health": [(200, {"status": "ok"})]})
        with unittest.mock.patch.object(server_manager, "pid_is_running", return_value=False):
            self.assertFalse(server_manager.wait_until_ready("127.0.0.1", server.port, 1, 15))

    def test_default_timeout_allows_slow_loads(self) -> None:
        self.assertGreaterEqual(server_manager.DEFAULT_READY_TIMEOUT_SECONDS, 120)

    def test_probe_capabilities_reads_props(self) -> None:
        props = {
            "default_generation_settings": {"n_ctx": 8192},
            "total_slots": 1,
            "modalities": {"vision": True, "video": False, "audio": False},
            "chat_template_caps": {"supports_tools": True, "supports_tool_calls": True},
            "build_info": "b6100-0e1f797",
        }
        server = self._serve({"/props": [(200, props)]})
        caps = server_manager.probe_capabilities("127.0.0.1", server.port)
        self.assertEqual(
            caps,
            {
                "slot_ctx": 8192,
                "total_slots": 1,
                "input_modalities": ["text", "image"],
                "tools": True,
                "build_info": "b6100-0e1f797",
            },
        )

    def test_probe_capabilities_absent_endpoint(self) -> None:
        server = self._serve({})
        self.assertIsNone(server_manager.probe_capabilities("127.0.0.1", server.port))


class ParsePropsTests(unittest.TestCase):
    def test_tools_need_both_template_caps(self) -> None:
        caps = server_manager.parse_props(
            {"chat_template_caps": {"supports_tools": True, "supports_tool_calls": False}, "chat_template": "tools"}
        )
        self.assertFalse(caps["tools"])

    def test_old_builds_fall_back_to_template_text(self) -> None:
        self.assertTrue(server_manager.parse_props({"chat_template": "{%- if tools %}x{% endif %}"})["tools"])
        self.assertFalse(server_manager.parse_props({"chat_template": "{{ message.content }}"})["tools"])

    def test_unknown_modalities_ignored_and_order_stable(self) -> None:
        caps = server_manager.parse_props(
            {"modalities": {"video": True, "some_future_modality": True, "audio": True, "vision": True}}
        )
        self.assertEqual(caps["input_modalities"], ["text", "image", "audio", "video"])

    def test_warns_when_slot_context_is_smaller_than_requested(self) -> None:
        caps = {"slot_ctx": 8192, "total_slots": 4}
        warnings = server_manager._capability_warnings(caps, 32768)
        self.assertEqual(len(warnings), 1)
        self.assertIn("8192", warnings[0])
        self.assertIn("4 parallel slots", warnings[0])
        self.assertEqual(server_manager._capability_warnings(caps, 8192), [])
        self.assertEqual(server_manager._capability_warnings(caps, None), [])


class ResponseMetricsTests(unittest.TestCase):
    def test_prefers_llama_server_timings(self) -> None:
        payload = {
            "usage": {"completion_tokens": 100, "prompt_tokens": 500},
            "timings": {
                "prompt_n": 480,
                "prompt_ms": 400.0,
                "prompt_per_second": 1200.0,
                "predicted_n": 100,
                "predicted_ms": 2000.0,
                "predicted_per_second": 50.0,
                "cache_n": 20,
            },
        }
        metrics = response_metrics(payload, "x" * 400, elapsed=4.0)
        self.assertEqual(metrics["tokens_per_second"], 50.0)
        self.assertEqual(metrics["tokens_per_second_source"], "timings")
        self.assertEqual(metrics["wall_tokens_per_second"], 25.0)
        self.assertEqual(metrics["prompt_tokens_per_second"], 1200.0)
        self.assertEqual(metrics["prompt_tokens"], 480)
        self.assertEqual(metrics["completion_tokens"], 100)
        self.assertEqual(metrics["cached_prompt_tokens"], 20)
        self.assertEqual(metrics["prompt_ms"], 400.0)
        self.assertEqual(metrics["generation_ms"], 2000.0)

    def test_falls_back_to_wall_clock_without_timings(self) -> None:
        metrics = response_metrics({"usage": {"completion_tokens": 60, "prompt_tokens": 12}}, "hi", elapsed=3.0)
        self.assertEqual(metrics["tokens_per_second"], 20.0)
        self.assertEqual(metrics["tokens_per_second_source"], "wall_clock")
        self.assertEqual(metrics["prompt_tokens"], 12)
        self.assertNotIn("prompt_tokens_per_second", metrics)

    def test_estimates_tokens_from_text_when_usage_missing(self) -> None:
        metrics = response_metrics({}, "a" * 40, elapsed=1.0)
        self.assertEqual(metrics["completion_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
