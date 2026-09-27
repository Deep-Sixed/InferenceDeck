from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

from inferencedeck.auth import AuthState
from inferencedeck.gateway import anthropic_api, openai_api, router
from inferencedeck.gateway.engines import OpenAICompatibleEngine
from inferencedeck.gateway.ir import GatewayError, StreamEvent, Usage
from inferencedeck.gateway.router import Target
from inferencedeck.gateway.server import make_server


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
        self.start_gateway(AuthState(token=""))

    def start_gateway(self, auth: AuthState) -> None:
        def resolve() -> Target:
            if self.target is None:
                raise GatewayError(503, "no inference target", "overloaded")
            return self.target
        gateway = make_server("127.0.0.1", 0, auth, resolve)
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

    def test_non_loopback_bind_needs_token(self) -> None:
        with self.assertRaises(Exception):
            make_server("0.0.0.0", 0, AuthState(token=""))


class RouterTests(unittest.TestCase):
    def test_remote_endpoint_wins_and_carries_key(self) -> None:
        remote = mock.Mock(valid=True, api_key_env="OR_KEY", key_required=True, display_name="OpenRouter model",
                           summary="OpenRouter · Cloud", base_url="https://openrouter.ai/api/v1", model="vendor/model")
        remote.name = "openrouter"
        with mock.patch.object(router, "active_endpoint", return_value=remote), \
                mock.patch.dict(os.environ, {"OR_KEY": "k"}):
            target = router.resolve_target()
        self.assertEqual(target.engine.api_base, "https://openrouter.ai/api/v1")
        self.assertEqual((target.engine.api_key, target.engine.model), ("k", "vendor/model"))
        self.assertEqual(target.label, "OpenRouter model (OpenRouter · Cloud)")

    def test_keyless_self_hosted_endpoint(self) -> None:
        remote = mock.Mock(valid=True, api_key_env="", key_required=False, display_name="Qwen",
                           summary="Thanatos · Tailscale · Self-hosted", base_url="http://thanatos:8080", model="")
        remote.name = "thanatos"
        with mock.patch.object(router, "active_endpoint", return_value=remote):
            target = router.resolve_target()
        self.assertEqual(target.engine.api_base, "http://thanatos:8080/v1")
        self.assertEqual(target.engine.api_key, "")

    def test_local_server_skips_paused(self) -> None:
        servers = [{"mode": "paused", "running": True, "suspended": True, "host": "127.0.0.1", "port": 1},
                   {"mode": "qwen", "running": True, "suspended": False, "host": "0.0.0.0", "port": 8080}]
        with mock.patch.object(router, "active_endpoint", return_value=None), \
                mock.patch.object(router, "list_servers", return_value=servers):
            target = router.resolve_target()
        self.assertEqual(target.engine.api_base, "http://127.0.0.1:8080/v1")
        self.assertEqual(target.model_id, "qwen")

    def test_nothing_running(self) -> None:
        with mock.patch.object(router, "active_endpoint", return_value=None), \
                mock.patch.object(router, "list_servers", return_value=[]):
            with self.assertRaises(GatewayError) as ctx:
                router.resolve_target()
        self.assertEqual(ctx.exception.status, 503)


if __name__ == "__main__":
    unittest.main()
