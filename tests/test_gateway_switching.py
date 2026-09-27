from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any
from unittest import mock

from inferencedeck.auth import AuthState
from inferencedeck.gateway import router
from inferencedeck.gateway.ir import GatewayError
from inferencedeck.gateway.server import make_server
from inferencedeck.gateway.switching import ModelSwitcher
from inferencedeck.tray import ApiError

from test_gateway import _FakeUpstream

PROFILES = [
    {"mode": "qwen", "name": "Qwen3 32B", "launchable": True, "params": {"alias": "qwen3-32b"}},
    {"mode": "llama", "name": "Llama 3.3 70B", "launchable": True, "params": {"alias": "llama-3.3-70b"}},
    {"mode": "broken", "name": "Missing model", "launchable": False, "params": {}},
]


def server(mode: str, port: int = 8080, **extra: Any) -> dict[str, Any]:
    return {"id": f"{mode}-id", "mode": mode, "running": True, "suspended": False,
            "host": "127.0.0.1", "port": port, **extra}


class FakeControlApi:
    """Stands in for inferencedeck-web: records calls and mutates a server list."""

    def __init__(self, servers: list[dict[str, Any]], port: int = 8080) -> None:
        self.servers = servers
        self.port = port
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: str | None = None

    def profiles(self) -> list[dict[str, Any]]:
        if self.fail == "profiles":
            raise ApiError("InferenceDeck is not reachable at http://127.0.0.1:8716")
        return PROFILES

    def request(self, path: str, body: dict[str, Any] | None = None, timeout: float = 5) -> dict[str, Any]:
        action = path.rsplit("/", 1)[-1]
        self.calls.append((action, dict(body or {})))
        if self.fail == action:
            return {"success": False, "error": "port in use"}
        if action == "release":
            for s in self.servers:
                if s["id"] == body["server_id"]:
                    s.update(running=False, status="parked")
            return {"success": True}
        if action in ("start", "restore", "resume"):
            mode = body.get("mode") or next(s["mode"] for s in self.servers if s["id"] == body["server_id"])
            self.servers[:] = [s for s in self.servers if s["mode"] != mode]
            started = server(mode, self.port)
            self.servers.append(started)
            return {"success": True, "server": started}
        raise AssertionError(f"unexpected call {path}")

    def list(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self.servers]


class RoutingTests(unittest.TestCase):
    def resolve(self, servers, model, switcher=None):
        with mock.patch.object(router, "active_endpoint", return_value=None), \
                mock.patch.object(router, "list_servers", return_value=servers):
            return router.resolve_target(model, switcher)

    def test_model_picks_the_running_server_it_names(self) -> None:
        servers = [server("qwen", 8080), server("llama", 8081)]
        self.assertEqual(self.resolve(servers, "llama").engine.api_base, "http://127.0.0.1:8081/v1")
        self.assertEqual(self.resolve(servers, "LLAMA").model_id, "llama")

    def test_unknown_model_falls_back_to_the_running_server(self) -> None:
        target = self.resolve([server("qwen")], "gpt-4o")
        self.assertEqual(target.model_id, "qwen")

    def test_without_switching_nothing_is_started(self) -> None:
        with self.assertRaises(GatewayError) as ctx:
            self.resolve([], "llama")
        self.assertEqual(ctx.exception.status, 503)


class SwitcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeControlApi([server("qwen")])
        self.switcher = ModelSwitcher(self.api, drain_timeout=2)

    def resolve(self, model):
        with mock.patch.object(router, "active_endpoint", return_value=None), \
                mock.patch.object(router, "list_servers", side_effect=self.api.list):
            return router.resolve_target(model, self.switcher)

    def test_named_profile_is_loaded_by_alias_after_releasing_the_other(self) -> None:
        target = self.resolve("llama-3.3-70b")
        self.assertEqual(target.model_id, "llama")
        self.assertEqual(self.api.calls, [("release", {"server_id": "qwen-id"}), ("start", {"mode": "llama"})])

    def test_loaded_model_is_used_without_calls(self) -> None:
        self.assertEqual(self.resolve("Qwen3 32B").model_id, "qwen")
        self.assertEqual(self.api.calls, [])

    def test_parked_profile_is_restored_and_paused_is_resumed(self) -> None:
        self.api.servers.append({**server("llama", 8081), "running": False, "status": "parked"})
        self.resolve("llama")
        self.assertEqual(self.api.calls[-1], ("restore", {"server_id": "llama-id"}))

        api = FakeControlApi([server("qwen"), server("llama", suspended=True)])
        self.api, self.switcher = api, ModelSwitcher(api)
        self.resolve("llama")
        self.assertEqual(api.calls, [("release", {"server_id": "qwen-id"}), ("resume", {"server_id": "llama-id"})])

    def test_unlaunchable_or_unknown_model_does_not_switch(self) -> None:
        self.assertEqual(self.resolve("broken").model_id, "qwen")
        self.assertEqual(self.resolve("gpt-4o").model_id, "qwen")
        self.assertEqual(self.api.calls, [])

    def test_failures_are_503s_that_name_the_step(self) -> None:
        self.api.fail = "start"
        with self.assertRaises(GatewayError) as ctx:
            self.resolve("llama")
        self.assertEqual(ctx.exception.status, 503)
        self.assertIn("could not start llama: port in use", str(ctx.exception))

        api = FakeControlApi([server("qwen")])
        api.fail = "profiles"
        self.switcher = ModelSwitcher(api)
        with self.assertRaises(GatewayError) as ctx:
            self.resolve("llama")
        self.assertIn("needs inferencedeck-web", str(ctx.exception))

    def test_switch_waits_for_in_flight_requests(self) -> None:
        released_at: list[float] = []
        original = self.api.request

        def request(path, body=None, timeout=5):
            if path.endswith("release"):
                released_at.append(time.monotonic())
            return original(path, body, timeout)

        self.api.request = request
        lease = self.switcher.inflight.lease("qwen-id")
        lease.__enter__()
        finished_at: list[float] = []

        def finish():
            time.sleep(0.3)
            finished_at.append(time.monotonic())
            lease.__exit__(None, None, None)

        threading.Thread(target=finish).start()
        self.resolve("llama")
        self.assertTrue(released_at and finished_at)
        self.assertGreaterEqual(released_at[0], finished_at[0])

    def test_concurrent_requests_switch_once(self) -> None:
        results: list[str] = []
        threads = [threading.Thread(target=lambda: results.append(self.resolve("llama").model_id)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, ["llama"] * 4)
        self.assertEqual([c[0] for c in self.api.calls].count("start"), 1)

    def test_models_lists_loaded_and_switchable_profiles(self) -> None:
        with mock.patch.object(router, "active_endpoint", return_value=None), \
                mock.patch.object(router, "list_servers", side_effect=self.api.list):
            models = router.list_models(self.switcher)
        self.assertEqual([(m["id"], m["loaded"]) for m in models], [("qwen", True), ("llama", False)])


class SwitchingGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeUpstream.received, _FakeUpstream.headers_seen, _FakeUpstream.reply_tool = [], [], False
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        port = upstream.server_address[1]
        self.api = FakeControlApi([server("qwen", port)], port=port)
        self.switcher = ModelSwitcher(self.api)
        patches = [mock.patch.object(router, "active_endpoint", return_value=None),
                   mock.patch.object(router, "list_servers", side_effect=self.api.list)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        gateway = make_server("127.0.0.1", 0, AuthState(token=""), switcher=self.switcher)
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    def post(self, body: dict[str, Any]) -> tuple[int, bytes]:
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read()

    def test_request_for_another_model_switches_then_answers(self) -> None:
        status, raw = self.post({"model": "llama", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["model"], "llama")
        self.assertEqual([c[0] for c in self.api.calls], ["release", "start"])

    def test_streamed_request_releases_its_lease(self) -> None:
        status, raw = self.post({"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertIn(b"[DONE]", raw)
        self.assertEqual(self.switcher.inflight.count("qwen-id"), 0)

    def test_models_endpoint(self) -> None:
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as response:
            data = json.loads(response.read())["data"]
        self.assertEqual([(m["id"], m["loaded"]) for m in data], [("qwen", True), ("llama", False)])


if __name__ == "__main__":
    unittest.main()
