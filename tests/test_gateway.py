from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from inferencedeck.auth import AuthState
from inferencedeck.gateway import anthropic_api, ollama_api, openai_api, router
from inferencedeck.gateway.engines import AnthropicEngine, OllamaEngine, OpenAICompatibleEngine
from inferencedeck.gateway.ir import GatewayError, StreamEvent, Usage
from inferencedeck.gateway.router import Target
from inferencedeck.remotes import list_endpoints
from inferencedeck.gateway.server import make_server
from inferencedeck.inflight import Tracker


def _sse_events(raw: bytes) -> list[tuple[str, Any]]:
    """(event name, parsed data) pairs from an SSE body; name is "" when absent."""
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data:
            events.append((name, data if data == "[DONE]" else json.loads(data)))
    return events


class _StubRouter:
    """Router stand-in: every request goes to whatever ``pick()`` returns."""

    def __init__(self, pick) -> None:
        self.pick = pick
        self.requested: list[str] = []

    def resolve(self, model: str = "") -> Target:
        self.requested.append(model)
        target = self.pick()
        if target is None:
            raise GatewayError(503, "no inference target", "overloaded")
        return target

    def catalog(self) -> list[Target]:
        target = self.pick()
        return [target] if target is not None else []


def _write_endpoints(root: Path, **configs: dict[str, Any]) -> None:
    for name, config in configs.items():
        (root / f"{name}.json").write_text(json.dumps(config))


class OpenAIAdapterTests(unittest.TestCase):
    def test_parse_and_rewire_round_trip(self) -> None:
        body = {
            "model": "qwen",
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
            ],
            "max_completion_tokens": 64, "stop": "END", "top_k": 20, "min_p": 0.05,
            "repeat_penalty": 1.1,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
        }
        request = openai_api.parse_request(body)
        self.assertEqual(request.messages[0].role, "system")
        self.assertEqual(request.sampling.max_tokens, 64)
        self.assertEqual(request.sampling.stop, ["END"])
        self.assertEqual(request.tool_choice, "f")
        self.assertEqual(request.extra, {"repeat_penalty": 1.1})
        wire = openai_api.to_wire(request, "served-name")
        self.assertEqual(wire["model"], "served-name")
        self.assertEqual(wire["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(wire["messages"][1]["content"][1]["image_url"]["url"], "data:image/png;base64,AA")
        self.assertIsNone(wire["messages"][2]["content"])
        self.assertEqual(wire["messages"][3]["tool_call_id"], "c1")
        self.assertEqual((wire["top_k"], wire["min_p"], wire["repeat_penalty"]), (20, 0.05, 1.1))
        self.assertEqual(wire["tool_choice"], {"type": "function", "function": {"name": "f"}})

    def test_rejects_bad_requests(self) -> None:
        for body in ({"messages": []}, {"messages": [{"role": "robot", "content": "x"}]},
                     {"messages": [{"role": "user", "content": "x"}], "n": 2}):
            with self.assertRaises(GatewayError) as ctx:
                openai_api.parse_request(body)
            self.assertEqual(ctx.exception.status, 400)

    def test_stream_events_from_sse(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":"He"}}]}\n', b"\n",
            b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\\"a\\""}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}\n',
            b"data: [DONE]\n",
        ]
        events = list(openai_api.stream_events(lines))
        self.assertEqual([e.kind for e in events], ["text", "text", "tool_call", "tool_call", "usage", "finish"])
        self.assertEqual(events[2].tool_name, "f")
        self.assertEqual(events[4].usage, Usage(5, 7))
        self.assertEqual(events[-1].finish_reason, "tool_calls")


class AnthropicAdapterTests(unittest.TestCase):
    def test_parse_tool_round_trip(self) -> None:
        request = anthropic_api.parse_request({
            "model": "claude-alias", "max_tokens": 100, "system": [{"type": "text", "text": "sys"}],
            "stop_sequences": ["X"], "top_k": 5, "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [{"type": "text", "text": "checking"},
                                                  {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "rain"},
                                             {"type": "text", "text": "thanks"}]},
            ],
        })
        roles = [m.role for m in request.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(json.loads(request.messages[2].tool_calls[0].arguments), {"city": "Oslo"})
        self.assertEqual(request.messages[3].tool_call_id, "t1")
        self.assertEqual(request.tool_choice, "required")
        self.assertEqual((request.sampling.max_tokens, request.sampling.top_k, request.sampling.stop), (100, 5, ["X"]))
        self.assertEqual(request.extra, {})  # thinking is dropped, never forwarded

    def test_max_tokens_required_and_server_tools_rejected(self) -> None:
        with self.assertRaises(GatewayError):
            anthropic_api.parse_request({"messages": [{"role": "user", "content": "x"}]})
        with self.assertRaises(GatewayError):
            anthropic_api.parse_request({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}],
                                         "tools": [{"type": "web_search_20250305", "name": "web_search"}]})

    def test_render_stream_text_then_tool(self) -> None:
        events = [
            StreamEvent("text", text="Hi"),
            StreamEvent("tool_call", index=0, tool_id="c1", tool_name="f", arguments='{"a"'),
            StreamEvent("tool_call", index=0, arguments=":1}"),
            StreamEvent("usage", usage=Usage(3, 4)),
            StreamEvent("finish", finish_reason="tool_calls"),
        ]
        out = _sse_events(b"".join(anthropic_api.render_stream(events, "m")))
        names = [name for name, _ in out]
        self.assertEqual(names, ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                                 "content_block_start", "content_block_delta", "content_block_delta",
                                 "content_block_stop", "message_delta", "message_stop"])
        self.assertEqual(out[4][1]["content_block"]["name"], "f")
        partial = "".join(d["delta"]["partial_json"] for n, d in out if n == "content_block_delta" and d["index"] == 1)
        self.assertEqual(json.loads(partial), {"a": 1})
        self.assertEqual(out[8][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(out[8][1]["usage"]["output_tokens"], 4)


class _FakeUpstream(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server; records what the gateway sent."""

    received: list[dict[str, Any]] = []
    headers_seen: list[dict[str, str]] = []
    reply_tool = False

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append(body)
        type(self).headers_seen.append(dict(self.headers))
        if body.get("model") == "missing":
            payload = json.dumps({"error": {"message": "model not found"}}).encode()
            self.send_response(404); self.send_header("Content-Length", str(len(payload))); self.end_headers()
            self.wfile.write(payload)
            return
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for chunk in ({"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
                          {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                          {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        message: dict[str, Any] = {"role": "assistant", "content": "Hello"}
        finish = "stop"
        if type(self).reply_tool:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'}}]}
            finish = "tool_calls"
        payload = json.dumps({"id": "up-1", "model": body.get("model"), "choices": [
            {"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload))); self.end_headers()
        self.wfile.write(payload)


class GatewayServerTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeUpstream.received, _FakeUpstream.headers_seen, _FakeUpstream.reply_tool = [], [], False
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.upstream_base = f"http://127.0.0.1:{self.upstream.server_address[1]}/v1"
        self.target: Target | None = Target(OpenAICompatibleEngine(self.upstream_base, api_key="up-key"),
                                            "Qwen (Thanatos · Tailscale · Self-hosted)", "qwen")
        counts = tempfile.TemporaryDirectory()
        self.addCleanup(counts.cleanup)
        self.inflight = Tracker(path=Path(counts.name) / "gateway.json")
        self.start_gateway(AuthState(token=""))

    def start_gateway(self, auth: AuthState) -> None:
        gateway = make_server("127.0.0.1", 0, auth, _StubRouter(lambda: self.target), inflight=self.inflight)
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    def post(self, path: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[int, bytes]:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _broken_stream_target(self) -> Target:
        class BrokenEngine:
            api_base = "http://upstream.invalid/v1"

            def stream(self, request):
                def events():
                    yield StreamEvent("text", text="Hel")
                    # Not a GatewayError: e.g. an upstream chunk of an unexpected shape.
                    raise AttributeError("'str' object has no attribute 'get'")
                return events()

        return Target(BrokenEngine(), "broken", "m", server_id="broken-1")

    def test_unexpected_stream_failure_is_reported_in_band(self) -> None:
        chat = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        anthropic = {"model": "m", "stream": True, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
        for path, body in (("/v1/chat/completions", chat), ("/v1/messages", anthropic)):
            with self.subTest(path=path):
                self.target = self._broken_stream_target()
                status, raw = self.post(path, body)
                self.assertEqual(status, 200)
                # One response only: no second status line written into the stream.
                self.assertNotIn(b"HTTP/1.", raw)
                self.assertIn(b"Hel", raw)
                self.assertIn(b"upstream stream failed", raw)
                # No longer counted, so a model switch or release isn't left waiting on it.
                self.assertEqual(self.inflight.counts()["broken-1"]["in_flight"], 0)

    def test_trusted_proxy_throttles_each_gateway_client_separately(self) -> None:
        self.start_gateway(AuthState(token="secret", trusted_proxies=frozenset({"127.0.0.1"})))
        body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}

        def guess(client: str) -> int:
            return self.post("/v1/chat/completions", body, {"Authorization": "Bearer wrong", "X-Forwarded-For": client})[0]

        for _ in range(5):
            self.assertEqual(guess("198.51.100.7"), 401)
        self.assertEqual(guess("198.51.100.7"), 429)
        self.assertEqual(guess("198.51.100.8"), 401)  # another user behind the proxy is not locked out

    def test_untrusted_forwarded_for_cannot_dodge_the_gateway_throttle(self) -> None:
        self.start_gateway(AuthState(token="secret", trusted_proxies=frozenset()))
        body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
        for i in range(5):
            self.post("/v1/chat/completions", body, {"Authorization": "Bearer wrong", "X-Forwarded-For": f"198.51.100.{i}"})
        status, _raw = self.post("/v1/chat/completions", body,
                                 {"Authorization": "Bearer wrong", "X-Forwarded-For": "198.51.100.99"})
        self.assertEqual(status, 429)

    def test_cross_site_style_posts_never_reach_upstream(self) -> None:
        # What a web page can send without a CORS preflight: text/plain, or a form.
        chat = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
        anthropic = {"model": "qwen", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
        for path, body in (("/v1/chat/completions", chat), ("/v1/messages", anthropic)):
            for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
                with self.subTest(path=path, content_type=content_type):
                    status, raw = self.post(path, body, {"Content-Type": content_type})
                    self.assertEqual(status, 415)
                    self.assertIn("application/json", raw.decode())
        # urllib's default for a body with no Content-Type is the form type.
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(chat).encode(), method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(ctx.exception.code, 415)
        self.assertEqual(_FakeUpstream.received, [])

    def test_json_with_a_charset_is_accepted(self) -> None:
        status, _raw = self.post("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                                 {"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(status, 200)

    def test_missing_token_is_401_before_the_content_type_check(self) -> None:
        self.start_gateway(AuthState(token="secret"))
        status, _raw = self.post("/v1/chat/completions", {"model": "qwen", "messages": []}, {"Content-Type": "text/plain"})
        self.assertEqual(status, 401)

    def test_openai_request_is_forwarded_with_server_side_key(self) -> None:
        status, raw = self.post("/v1/chat/completions", {"model": "qwen", "messages": [{"role": "user", "content": "hi"}],
                                                         "min_p": 0.1}, {"Authorization": "Bearer client-key"})
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(payload["usage"]["total_tokens"], 4)
        self.assertEqual(_FakeUpstream.received[0]["min_p"], 0.1)
        # The upstream sees the gateway's key, never the client's.
        self.assertEqual(_FakeUpstream.headers_seen[0]["Authorization"], "Bearer up-key")

    def test_anthropic_request_to_openai_engine_with_tool_call(self) -> None:
        _FakeUpstream.reply_tool = True
        status, raw = self.post("/v1/messages", {
            "model": "claude-alias", "max_tokens": 50, "system": "sys",
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "weather in Oslo?"}],
        })
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["model"], "claude-alias")
        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"], [{"type": "tool_use", "id": "call_1", "name": "get_weather",
                                               "input": {"city": "Oslo"}}])
        sent = _FakeUpstream.received[0]
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(sent["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(sent["max_tokens"], 50)

    def test_anthropic_streaming_from_openai_engine(self) -> None:
        status, raw = self.post("/v1/messages", {"model": "m", "max_tokens": 10, "stream": True,
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        events = _sse_events(raw)
        text = "".join(d["delta"]["text"] for n, d in events if n == "content_block_delta")
        self.assertEqual(text, "Hello")
        self.assertEqual(events[-1][0], "message_stop")
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 2)
        self.assertTrue(_FakeUpstream.received[0]["stream_options"]["include_usage"])

    def test_openai_streaming_usage_only_when_asked(self) -> None:
        for include, expect in ((False, False), (True, True)):
            body: dict[str, Any] = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            if include:
                body["stream_options"] = {"include_usage": True}
            status, raw = self.post("/v1/chat/completions", body)
            self.assertEqual(status, 200)
            events = [d for _, d in _sse_events(raw)]
            self.assertEqual(events[-1], "[DONE]")
            self.assertEqual(any(isinstance(d, dict) and d.get("usage") for d in events), expect)
            text = "".join((d["choices"][0]["delta"].get("content") or "") for d in events
                           if isinstance(d, dict) and d.get("choices"))
            self.assertEqual(text, "Hello")

    def test_errors_use_the_callers_format(self) -> None:
        status, raw = self.post("/v1/messages", {"model": "missing", "max_tokens": 5,
                                                 "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(raw)["type"], "error")
        self.assertIn("model not found", json.loads(raw)["error"]["message"])
        self.target = None
        status, raw = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 503)
        self.assertIn("no inference target", json.loads(raw)["error"]["message"])

    def test_upstream_unreachable_is_502(self) -> None:
        self.target = Target(OpenAICompatibleEngine("http://127.0.0.1:9/v1", timeout=2), "dead", "dead")
        status, raw = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 502)

    def test_models_lists_current_target(self) -> None:
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["data"][0]["id"], "qwen")

    def test_token_required_when_configured(self) -> None:
        self.start_gateway(AuthState(token="s3cret"))
        body = {"messages": [{"role": "user", "content": "x"}]}
        self.assertEqual(self.post("/v1/chat/completions", body)[0], 401)
        self.assertEqual(self.post("/v1/chat/completions", body, {"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.post("/v1/chat/completions", body, {"Authorization": "Bearer s3cret"})[0], 200)
        status, _ = self.post("/v1/messages", {"max_tokens": 5, **body}, {"x-api-key": "s3cret"})
        self.assertEqual(status, 200)

    def test_trusted_proxy_throttles_by_forwarded_client(self) -> None:
        self.start_gateway(AuthState(token="s3cret", trusted_proxies=frozenset({"127.0.0.1"})))
        body = {"messages": [{"role": "user", "content": "x"}]}
        attacker = {"Authorization": "Bearer wrong", "X-Forwarded-For": "203.0.113.5"}
        for _ in range(5):
            self.assertEqual(self.post("/v1/chat/completions", body, attacker)[0], 401)
        self.assertEqual(self.post("/v1/chat/completions", body, attacker)[0], 429)
        # Another client behind the same proxy has its own bucket.
        other = {"Authorization": "Bearer s3cret", "X-Forwarded-For": "198.51.100.7"}
        self.assertEqual(self.post("/v1/chat/completions", body, other)[0], 200)

    def test_forwarded_for_ignored_from_untrusted_peer(self) -> None:
        self.start_gateway(AuthState(token="s3cret", trusted_proxies=frozenset()))
        body = {"messages": [{"role": "user", "content": "x"}]}
        for n in range(5):
            headers = {"Authorization": "Bearer wrong", "X-Forwarded-For": f"203.0.113.{n}"}
            self.assertEqual(self.post("/v1/chat/completions", body, headers)[0], 401)
        # Rotating the header doesn't dodge the throttle: the peer address is the key.
        fresh = {"Authorization": "Bearer s3cret", "X-Forwarded-For": "198.51.100.7"}
        status, raw = self.post("/v1/chat/completions", body, fresh)
        self.assertEqual(status, 429)
        self.assertIn("too many failed attempts", json.loads(raw)["error"]["message"])

    def test_valid_token_clears_earlier_failures(self) -> None:
        self.start_gateway(AuthState(token="s3cret"))
        body = {"messages": [{"role": "user", "content": "x"}]}
        for _ in range(4):
            self.assertEqual(self.post("/v1/chat/completions", body, {"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.post("/v1/chat/completions", body, {"Authorization": "Bearer s3cret"})[0], 200)
        # One more typo later doesn't trip the lockout.
        self.assertEqual(self.post("/v1/chat/completions", body, {"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.post("/v1/chat/completions", body, {"Authorization": "Bearer s3cret"})[0], 200)

    def test_client_forwarded_for_line_cannot_dodge_the_throttle(self) -> None:
        # A proxy that adds its own X-Forwarded-For line (HAProxy) leaves the
        # client's forged line first; the throttle must key on the proxy's.
        self.start_gateway(AuthState(token="s3cret", trusted_proxies=frozenset({"127.0.0.1"})))
        port = int(self.base.rsplit(":", 1)[1])
        payload = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()

        def guess(forged: str) -> int:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.putrequest("POST", "/v1/chat/completions")
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(len(payload)))
                conn.putheader("Authorization", "Bearer wrong")
                conn.putheader("X-Forwarded-For", forged)
                conn.putheader("X-Forwarded-For", "203.0.113.5")
                conn.endheaders(payload)
                return conn.getresponse().status
            finally:
                conn.close()

        for n in range(5):
            self.assertEqual(guess(f"10.9.9.{n}"), 401)
        self.assertEqual(guess("10.9.9.99"), 429)

    def test_non_loopback_bind_needs_token(self) -> None:
        with self.assertRaises(Exception):
            make_server("0.0.0.0", 0, AuthState(token=""))


class NameRoutingGatewayTests(unittest.TestCase):
    def _upstream(self) -> int:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def setUp(self) -> None:
        _FakeUpstream.received, _FakeUpstream.headers_seen, _FakeUpstream.reply_tool = [], [], False
        self.big, self.small = self._upstream(), self._upstream()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        _write_endpoints(root,
                         thanatos={"provider": "llamacpp", "enabled": True, "model": "qwen3-32b",
                                   "aliases": ["big"], "baseUrl": f"http://127.0.0.1:{self.big}"},
                         friday={"provider": "llamacpp", "model": "qwen3-4b", "aliases": ["small"],
                                 "baseUrl": f"http://127.0.0.1:{self.small}/v1"})
        gateway = make_server("127.0.0.1", 0, AuthState(token=""),
                              router.Router(endpoints=lambda: list_endpoints(root), servers=lambda: []))
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    post = GatewayServerTests.post

    def test_requests_reach_the_named_target(self) -> None:
        for name, port, upstream_model in (("small", self.small, "qwen3-4b"), ("big", self.big, "qwen3-32b"),
                                           ("unknown", self.big, "qwen3-32b")):
            status, raw = self.post("/v1/chat/completions", {"model": name, "messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(raw)["model"], name)  # the client sees the name it asked for
            self.assertTrue(_FakeUpstream.headers_seen[-1]["Host"].endswith(f":{port}"), name)
            self.assertEqual(_FakeUpstream.received[-1]["model"], upstream_model)

    def test_models_lists_every_routable_target(self) -> None:
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as response:
            data = json.loads(response.read())["data"]
        self.assertEqual([(m["id"], m["aliases"], m["default"]) for m in data],
                         [("big", ["qwen3-32b", "thanatos"], True), ("small", ["qwen3-4b", "friday"], False)])


class RouterTests(unittest.TestCase):
    """Routing over real endpoint configs and stubbed local servers."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.servers: list[dict[str, Any]] = []
        self.router = router.Router(endpoints=lambda: list_endpoints(self.root), servers=lambda: self.servers)

    def test_enabled_remote_is_default_and_carries_key(self) -> None:
        _write_endpoints(self.root, openrouter={
            "provider": "openrouter", "lane": "true_cloud", "enabled": True, "model": "vendor/model",
            "baseUrl": "https://openrouter.ai/api/v1", "apiKeyEnv": "OR_KEY", "displayName": "OpenRouter model"})
        with mock.patch.dict(os.environ, {"OR_KEY": "k"}):
            target = self.router.resolve()
        self.assertEqual(target.engine.api_base, "https://openrouter.ai/api/v1")
        self.assertEqual((target.engine.api_key, target.engine.model), ("k", "vendor/model"))
        self.assertEqual(target.label, "OpenRouter model (OpenRouter · Cloud)")
        self.assertTrue(target.default)

    def test_enabled_remote_missing_key_is_an_error_not_a_fallback(self) -> None:
        _write_endpoints(self.root, openrouter={
            "provider": "openrouter", "lane": "true_cloud", "enabled": True,
            "baseUrl": "https://openrouter.ai/api/v1", "apiKeyEnv": "UNSET_ROUTER_TEST_KEY"})
        self.servers = [{"mode": "qwen", "running": True, "host": "127.0.0.1", "port": 8080}]
        os.environ.pop("UNSET_ROUTER_TEST_KEY", None)
        with self.assertRaises(GatewayError) as ctx:
            self.router.resolve("qwen")
        self.assertEqual(ctx.exception.status, 503)

    def test_keyless_self_hosted_endpoint(self) -> None:
        _write_endpoints(self.root, thanatos={"provider": "llamacpp", "enabled": True,
                                              "baseUrl": "http://thanatos:8080"})
        target = self.router.resolve()
        self.assertEqual((target.engine.api_base, target.engine.api_key), ("http://thanatos:8080/v1", ""))

    def test_local_server_skips_paused(self) -> None:
        self.servers = [{"mode": "paused", "running": True, "suspended": True, "host": "127.0.0.1", "port": 1},
                        {"mode": "qwen", "running": True, "suspended": False, "host": "0.0.0.0", "port": 8080}]
        target = self.router.resolve()
        self.assertEqual(target.engine.api_base, "http://127.0.0.1:8080/v1")
        self.assertEqual(target.model_id, "qwen")

    def test_nothing_running(self) -> None:
        with self.assertRaises(GatewayError) as ctx:
            self.router.resolve()
        self.assertEqual(ctx.exception.status, 503)

    def _estate(self) -> None:
        """A local server, a self-hosted box, a cloud endpoint opted in, and one not."""
        self.servers = [{"id": "qwen-1-x", "mode": "qwen", "running": True, "host": "127.0.0.1", "port": 8080,
                         "model_path": "/models/Qwen3-8B-Q4_K_M.gguf"}]
        _write_endpoints(
            self.root,
            thanatos={"provider": "llamacpp", "baseUrl": "http://thanatos:8080", "model": "qwen3-32b",
                      "aliases": ["big-qwen"]},
            openrouter={"provider": "openrouter", "lane": "true_cloud", "routable": True, "model": "vendor/default",
                        "baseUrl": "https://openrouter.ai/api/v1", "apiKeyEnv": "OR_KEY"},
            private_cloud={"provider": "openai", "lane": "true_cloud", "model": "gpt-private",
                           "baseUrl": "https://example.invalid/v1", "apiKeyEnv": "OR_KEY"},
        )

    def test_routes_by_name_across_targets(self) -> None:
        self._estate()
        with mock.patch.dict(os.environ, {"OR_KEY": "k"}):
            self.assertEqual(self.router.resolve("big-qwen").engine.api_base, "http://thanatos:8080/v1")
            self.assertEqual(self.router.resolve("QWEN3-32B").engine.api_base, "http://thanatos:8080/v1")
            self.assertEqual(self.router.resolve("Qwen3-8B-Q4_K_M").engine.api_base, "http://127.0.0.1:8080/v1")
            self.assertEqual(self.router.resolve("vendor/default").engine.api_base, "https://openrouter.ai/api/v1")
            # Unknown names and no name go to the default (the local server here).
            self.assertEqual(self.router.resolve("gpt-4o").engine.api_base, "http://127.0.0.1:8080/v1")
            self.assertEqual(self.router.resolve("").engine.api_base, "http://127.0.0.1:8080/v1")

    def test_cloud_endpoint_needs_opt_in(self) -> None:
        self._estate()
        with mock.patch.dict(os.environ, {"OR_KEY": "k"}):
            names = [t.model_id for t in self.router.catalog()]
            target = self.router.resolve("gpt-private")
        self.assertNotIn("gpt-private", names)
        self.assertEqual(target.engine.api_base, "http://127.0.0.1:8080/v1")

    def test_endpoint_prefix_picks_any_model_on_that_endpoint(self) -> None:
        self._estate()
        with mock.patch.dict(os.environ, {"OR_KEY": "k"}):
            target = self.router.resolve("openrouter/meta-llama/llama-3.3-70b-instruct")
            catalog_engine = next(t for t in self.router.catalog() if t.endpoint == "openrouter").engine
        self.assertEqual(target.engine.api_base, "https://openrouter.ai/api/v1")
        self.assertEqual(target.engine.model, "meta-llama/llama-3.3-70b-instruct")
        self.assertEqual(catalog_engine.model, "vendor/default")  # the catalog entry is untouched

    def test_routable_endpoint_without_key_is_skipped(self) -> None:
        self._estate()
        os.environ.pop("OR_KEY", None)
        names = [t.model_id for t in self.router.catalog()]
        self.assertNotIn("vendor/default", names)
        self.assertEqual(self.router.resolve("vendor/default").engine.api_base, "http://127.0.0.1:8080/v1")

    def test_enabled_remote_stays_default_but_others_route_by_name(self) -> None:
        self._estate()
        _write_endpoints(self.root, thanatos={"provider": "llamacpp", "baseUrl": "http://thanatos:8080",
                                              "model": "qwen3-32b", "enabled": True})
        catalog = self.router.catalog()
        self.assertTrue(catalog[0].default)
        self.assertEqual(catalog[0].endpoint, "thanatos")
        self.assertEqual(self.router.resolve("anything").engine.api_base, "http://thanatos:8080/v1")
        self.assertEqual(self.router.resolve("qwen").engine.api_base, "http://127.0.0.1:8080/v1")

    def test_self_hosted_can_opt_out(self) -> None:
        _write_endpoints(self.root, thanatos={"provider": "llamacpp", "baseUrl": "http://thanatos:8080",
                                              "model": "qwen3-32b", "routable": False})
        self.assertEqual(self.router.catalog(), [])


class OllamaAdapterTests(unittest.TestCase):
    def test_to_wire_maps_options_images_tools_and_format(self) -> None:
        request = openai_api.parse_request({
            "model": "client-name",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "found"},
            ],
            "max_tokens": 32, "temperature": 0.2, "top_k": 40, "stop": ["END"],
            "num_ctx": 8192, "keep_alive": "5m", "parallel_tool_calls": False,
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        })
        wire = ollama_api.to_wire(request, "qwen3:32b")
        self.assertEqual(wire["model"], "qwen3:32b")
        self.assertFalse(wire["stream"])
        self.assertEqual(wire["options"], {"num_predict": 32, "temperature": 0.2, "top_k": 40, "stop": ["END"],
                                           "num_ctx": 8192})
        self.assertEqual(wire["keep_alive"], "5m")
        self.assertNotIn("parallel_tool_calls", json.dumps(wire))
        self.assertEqual(wire["messages"][0]["images"], ["QUJD"])
        self.assertEqual(wire["messages"][1]["tool_calls"], [{"function": {"name": "lookup", "arguments": {"q": "x"}}}])
        self.assertEqual(wire["messages"][2], {"role": "tool", "content": "found", "tool_name": "lookup"})
        self.assertEqual(wire["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(wire["format"], {"type": "object"})

    def test_tool_choice_none_withholds_tools_and_image_urls_are_refused(self) -> None:
        request = openai_api.parse_request({
            "messages": [{"role": "user", "content": "hi"}], "tool_choice": "none",
            "tools": [{"type": "function", "function": {"name": "f"}}]})
        self.assertNotIn("tools", ollama_api.to_wire(request, "m"))
        request = openai_api.parse_request({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.invalid/cat.png"}}]}]})
        with self.assertRaises(GatewayError) as ctx:
            ollama_api.to_wire(request, "m")
        self.assertEqual(ctx.exception.status, 400)

    def test_from_wire_tool_calls_get_ids(self) -> None:
        result = ollama_api.from_wire({"model": "m", "done": True, "done_reason": "stop",
                                       "message": {"role": "assistant", "content": "",
                                                   "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
                                       "prompt_eval_count": 9, "eval_count": 4}, "m")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual((result.tool_calls[0].id, json.loads(result.tool_calls[0].arguments)), ("call_0", {"a": 1}))
        self.assertEqual(result.usage, Usage(9, 4))
        length = ollama_api.from_wire({"message": {"content": "cut"}, "done_reason": "length"}, "m")
        self.assertEqual(length.finish_reason, "length")

    def test_ndjson_stream(self) -> None:
        lines = [
            b'{"message":{"role":"assistant","content":"Hel"},"done":false}\n',
            b'{"message":{"role":"assistant","content":"lo"},"done":false}\n',
            b'{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{}}}]},"done":false}\n',
            b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}\n',
        ]
        events = list(ollama_api.stream_events(lines))
        self.assertEqual([e.kind for e in events], ["text", "text", "tool_call", "usage", "finish"])
        self.assertEqual(events[2].tool_id, "call_0")
        self.assertEqual(events[-1].finish_reason, "tool_calls")
        with self.assertRaises(GatewayError):
            list(ollama_api.stream_events([b'{"error":"model \\"x\\" not found"}\n']))


class _FakeOllama(BaseHTTPRequestHandler):
    received: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append({"path": self.path, "body": body})
        if body.get("model") == "missing":
            payload = json.dumps({"error": 'model "missing" not found, try pulling it first'}).encode()
            self.send_response(404); self.send_header("Content-Length", str(len(payload))); self.end_headers()
            self.wfile.write(payload)
            return
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
            for chunk in ({"message": {"role": "assistant", "content": "Hel"}, "done": False},
                          {"message": {"role": "assistant", "content": "lo"}, "done": False},
                          {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                           "prompt_eval_count": 2, "eval_count": 2}):
                self.wfile.write((json.dumps(chunk) + "\n").encode())
            return
        payload = json.dumps({"model": body["model"], "done": True, "done_reason": "stop",
                              "message": {"role": "assistant", "content": "Hello"},
                              "prompt_eval_count": 3, "eval_count": 1}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload))); self.end_headers()
        self.wfile.write(payload)


class OllamaGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeOllama.received = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        engine = OllamaEngine(f"http://127.0.0.1:{upstream.server_address[1]}", model="qwen3:32b")
        self.target = Target(engine, "Remote Ollama", "qwen3:32b")
        gateway = make_server("127.0.0.1", 0, AuthState(token=""), _StubRouter(lambda: self.target))
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    post = GatewayServerTests.post

    def test_openai_client_to_native_ollama(self) -> None:
        status, raw = self.post("/v1/chat/completions", {"model": "anything", "max_tokens": 8,
                                                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "Hello")
        sent = _FakeOllama.received[0]
        self.assertEqual(sent["path"], "/api/chat")
        self.assertEqual((sent["body"]["model"], sent["body"]["stream"]), ("qwen3:32b", False))
        self.assertEqual(sent["body"]["options"], {"num_predict": 8})

    def test_anthropic_streaming_from_ollama(self) -> None:
        status, raw = self.post("/v1/messages", {"model": "m", "max_tokens": 10, "stream": True,
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        events = _sse_events(raw)
        self.assertEqual("".join(d["delta"]["text"] for n, d in events if n == "content_block_delta"), "Hello")
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 2)
        self.assertEqual(events[-1][0], "message_stop")

    def test_ollama_error_reaches_client(self) -> None:
        self.target = Target(OllamaEngine(self.target.engine.api_base), "Remote Ollama", "missing")
        status, raw = self.post("/v1/chat/completions", {"model": "missing", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 404)
        self.assertIn("try pulling it first", json.loads(raw)["error"]["message"])


class OllamaRoutingTests(unittest.TestCase):
    def test_ollama_endpoint_uses_native_engine_at_server_root(self) -> None:
        for url in ("http://thanatos:11434", "http://thanatos:11434/v1", "http://thanatos:11434/api/"):
            with tempfile.TemporaryDirectory() as tmp:
                _write_endpoints(Path(tmp), ollama={"provider": "ollama", "enabled": True, "baseUrl": url,
                                                    "model": "qwen3:32b"})
                target = router.Router(endpoints=lambda: list_endpoints(Path(tmp)), servers=lambda: []).resolve()
            self.assertIsInstance(target.engine, OllamaEngine)
            self.assertEqual(target.engine.api_base, "http://thanatos:11434", url)

    def test_ollama_config_without_lane_is_self_hosted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "o.json").write_text(json.dumps({"provider": "ollama", "baseUrl": "http://thanatos:11434"}))
            cfg = list_endpoints(Path(tmp))[0]
        self.assertEqual(cfg.lane, "remote_host")
        self.assertTrue(cfg.valid, cfg.error)



class AnthropicEngineMappingTests(unittest.TestCase):
    def _request(self, **extra: Any):
        return openai_api.parse_request({
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "look"},
                                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
                                             {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call.1", "type": "function", "function": {"name": "a", "arguments": '{"x":1}'}},
                    {"id": "call.2", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call.1", "content": "one"},
                {"role": "tool", "tool_call_id": "call.2", "content": "two"},
                {"role": "user", "content": "and?"},
            ],
            "tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
                      {"type": "function", "function": {"name": "b"}}],
            **extra,
        })

    def test_to_wire_shapes_a_valid_messages_request(self) -> None:
        wire = anthropic_api.to_wire(self._request(
            temperature=0.3, min_p=0.1, seed=7, repeat_penalty=1.1, stop="END", tool_choice="required",
            response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}), "claude-opus-5")
        self.assertEqual(wire["system"], "be brief")
        self.assertEqual(wire["max_tokens"], anthropic_api.DEFAULT_MAX_TOKENS)
        self.assertEqual([m["role"] for m in wire["messages"]], ["user", "assistant", "user"])
        images = [b["source"] for b in wire["messages"][0]["content"] if b["type"] == "image"]
        self.assertEqual(images, [{"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
                                  {"type": "url", "url": "https://example.invalid/a.png"}])
        # Empty assistant text is dropped; ids are made API-safe and stay paired.
        self.assertEqual(wire["messages"][1]["content"], [
            {"type": "tool_use", "id": "call_1", "name": "a", "input": {"x": 1}},
            {"type": "tool_use", "id": "call_2", "name": "b", "input": {}}])
        # Parallel tool results and the text after them share one user turn.
        self.assertEqual(wire["messages"][2]["content"], [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "one"},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "two"},
            {"type": "text", "text": "and?"}])
        self.assertEqual(wire["tool_choice"], {"type": "any"})
        self.assertEqual(wire["stop_sequences"], ["END"])
        self.assertEqual(wire["output_config"], {"format": {"type": "json_schema", "schema": {"type": "object"}}})
        self.assertEqual(wire["temperature"], 0.3)
        # Fields the Messages API would reject are not sent.
        for field in ("min_p", "seed", "repeat_penalty", "stop", "response_format"):
            self.assertNotIn(field, wire)

    def test_streaming_defaults_to_a_larger_max_tokens(self) -> None:
        request = openai_api.parse_request({"messages": [{"role": "user", "content": "x"}], "stream": True})
        self.assertEqual(anthropic_api.to_wire(request, "m")["max_tokens"], anthropic_api.DEFAULT_STREAM_MAX_TOKENS)
        request = openai_api.parse_request({"messages": [{"role": "user", "content": "x"}], "max_tokens": 50})
        self.assertEqual(anthropic_api.to_wire(request, "m")["max_tokens"], 50)

    def test_from_wire_skips_thinking_and_totals_cached_input(self) -> None:
        result = anthropic_api.from_wire({
            "id": "msg_1", "model": "claude-opus-5", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                        {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "toolu_1", "name": "a", "input": {"x": 1}}],
            "usage": {"input_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10,
                      "output_tokens": 7}}, "m")
        self.assertEqual(result.text, "Checking.")
        self.assertEqual((result.tool_calls[0].id, json.loads(result.tool_calls[0].arguments)), ("toolu_1", {"x": 1}))
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.usage, Usage(115, 7))
        for reason, finish in (("max_tokens", "length"), ("refusal", "content_filter"),
                               ("model_context_window_exceeded", "length"), ("stop_sequence", "stop")):
            self.assertEqual(anthropic_api.from_wire({"stop_reason": reason, "content": []}, "m").finish_reason, finish)

    def test_stream_events(self) -> None:
        def sse(event: dict[str, Any]) -> list[bytes]:
            return [f"event: {event['type']}\n".encode(), f"data: {json.dumps(event)}\n".encode(), b"\n"]
        lines = [line for event in (
            {"type": "message_start", "message": {"usage": {"input_tokens": 4, "output_tokens": 1}}},
            {"type": "ping"},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_start", "index": 2,
             "content_block": {"type": "tool_use", "id": "toolu_9", "name": "a", "input": {}}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"x"'}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ":1}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
            {"type": "message_stop"},
        ) for line in sse(event)]
        events = list(anthropic_api.stream_events(lines))
        self.assertEqual([e.kind for e in events], ["text", "tool_call", "tool_call", "tool_call", "usage", "finish"])
        self.assertEqual((events[1].tool_id, events[1].tool_name, events[1].index), ("toolu_9", "a", 0))
        self.assertEqual(events[2].arguments + events[3].arguments, '{"x":1}')
        self.assertEqual(events[4].usage, Usage(4, 12))
        self.assertEqual(events[5].finish_reason, "tool_calls")
        with self.assertRaises(GatewayError) as ctx:
            list(anthropic_api.stream_events(sse({"type": "error",
                                                  "error": {"type": "overloaded_error", "message": "Overloaded"}})))
        self.assertEqual(ctx.exception.kind, "overloaded")


class _FakeAnthropic(BaseHTTPRequestHandler):
    received: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # Header names are case-insensitive (urllib sends "X-api-key"); compare lowercased.
        headers = {name.lower(): value for name, value in self.headers.items()}
        type(self).received.append({"path": self.path, "headers": headers, "body": body})
        if body["model"] == "busy":
            self._json(529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
            return
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for event in (
                {"type": "message_start", "message": {"id": "msg_s", "usage": {"input_tokens": 3, "output_tokens": 1}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
                {"type": "message_stop"},
            ):
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
            return
        self._json(200, {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                         "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                      "input": {"city": "Oslo"}}],
                         "stop_reason": "tool_use", "usage": {"input_tokens": 9, "output_tokens": 4}})


class AnthropicGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAnthropic.received = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAnthropic)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        self.engine = AnthropicEngine(f"http://127.0.0.1:{upstream.server_address[1]}", model="claude-opus-5",
                                      api_key="sk-ant-test")
        self.target = Target(self.engine, "Claude (Anthropic · Cloud)", "claude-opus-5")
        gateway = make_server("127.0.0.1", 0, AuthState(token=""), _StubRouter(lambda: self.target))
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    post = GatewayServerTests.post

    def test_openai_client_tool_call_through_anthropic(self) -> None:
        status, raw = self.post("/v1/chat/completions", {
            "model": "claude", "messages": [{"role": "user", "content": "weather in Oslo?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        }, {"Authorization": "Bearer client-key"})
        self.assertEqual(status, 200)
        choice = json.loads(raw)["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"city": "Oslo"})
        sent = _FakeAnthropic.received[0]
        self.assertEqual(sent["path"], "/v1/messages")
        self.assertEqual(sent["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(sent["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", sent["headers"])  # neither ours nor the client's
        self.assertEqual(sent["body"]["model"], "claude-opus-5")
        self.assertEqual(sent["body"]["tools"][0]["input_schema"], {"type": "object"})

    def test_openai_streaming_through_anthropic(self) -> None:
        status, raw = self.post("/v1/chat/completions", {"model": "claude", "stream": True,
                                                         "stream_options": {"include_usage": True},
                                                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        chunks = [d for _, d in _sse_events(raw) if isinstance(d, dict)]
        text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
        self.assertEqual(text, "Hello")
        self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    def test_anthropic_client_through_anthropic_engine(self) -> None:
        status, raw = self.post("/v1/messages", {"model": "claude", "max_tokens": 64, "system": "sys",
                                                 "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["content"][0]["input"], {"city": "Oslo"})
        self.assertEqual(_FakeAnthropic.received[0]["body"]["system"], "sys")
        self.assertEqual(_FakeAnthropic.received[0]["body"]["max_tokens"], 64)

    def test_overloaded_529_becomes_503(self) -> None:
        self.target = Target(AnthropicEngine(self.engine.api_base, model="busy"), "busy", "busy")
        status, raw = self.post("/v1/messages", {"model": "busy", "max_tokens": 5,
                                                 "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(raw)["error"]["type"], "overloaded_error")


class AnthropicRoutingTests(unittest.TestCase):
    def test_anthropic_endpoint_uses_messages_engine_at_api_root(self) -> None:
        for url in ("https://api.anthropic.com", "https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"):
            with tempfile.TemporaryDirectory() as tmp:
                _write_endpoints(Path(tmp), claude={"provider": "anthropic", "lane": "true_cloud", "enabled": True,
                                                    "baseUrl": url, "model": "claude-opus-5",
                                                    "apiKeyEnv": "ANTHROPIC_TEST_KEY"})
                with mock.patch.dict(os.environ, {"ANTHROPIC_TEST_KEY": "k"}):
                    target = router.Router(endpoints=lambda: list_endpoints(Path(tmp)), servers=lambda: []).resolve()
            self.assertIsInstance(target.engine, AnthropicEngine)
            self.assertEqual(target.engine.api_base, "https://api.anthropic.com", url)
            self.assertEqual(target.engine.api_key, "k")


if __name__ == "__main__":
    unittest.main()
