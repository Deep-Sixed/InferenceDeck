from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

from inferencedeck import inflight, model_switch
from inferencedeck.auth import AuthState
from inferencedeck.gateway.ir import GatewayError
from inferencedeck.gateway.router import Router
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
    """Stands in for inferencedeck-web: runs the real switch (model_switch) over a fake server table."""

    def __init__(self, servers: list[dict[str, Any]], port: int = 8080) -> None:
        self.servers = servers
        self.port = port
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail: str | None = None
        # Gateway requests running per server id; None reads the real in-flight files.
        self.busy: Callable[[], dict[str, int]] | None = lambda: {}

    def install(self, test: unittest.TestCase) -> None:
        """Point model_switch at this fake's server table for the rest of ``test``."""
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        patcher = mock.patch.multiple(
            model_switch.sm, list_servers=self.list, release_gpu=self._release, resume_server=self._resume,
            restore_server=self._restore, start_profile=self._start,
            state_path=lambda: Path(tmp.name) / "servers.json")
        patcher.start()
        test.addCleanup(patcher.stop)

    def profiles(self) -> list[dict[str, Any]]:
        if self.fail == "profiles":
            raise ApiError("InferenceDeck is not reachable at http://127.0.0.1:8716")
        return PROFILES

    def request(self, path: str, body: dict[str, Any] | None = None, timeout: float = 5) -> dict[str, Any]:
        action = path.rsplit("/", 1)[-1]
        if action != "switch":
            raise AssertionError(f"unexpected call {path}")
        body = dict(body or {})
        extra = {} if self.busy is None else {"busy": self.busy}
        return model_switch.switch_to(body["mode"], drain_timeout=body["drain_timeout"], **extra)

    def _record(self, action: str, body: dict[str, Any]) -> bool:
        self.calls.append((action, body))
        return self.fail != action

    def _release(self, server_id: str) -> dict[str, Any]:
        if not self._record("release", {"server_id": server_id}):
            return {"success": False, "error": "could not stop"}
        for s in self.servers:
            if s["id"] == server_id:
                s.update(running=False, status="parked")
        return {"success": True}

    def _come_up(self, mode: str) -> dict[str, Any]:
        self.servers[:] = [s for s in self.servers if s["mode"] != mode]
        started = server(mode, self.port)
        self.servers.append(started)
        return {"success": True, "server": started}

    def _resume(self, server_id: str) -> dict[str, Any]:
        self._record("resume", {"server_id": server_id})
        return self._come_up(next(s["mode"] for s in self.servers if s["id"] == server_id))

    def _restore(self, server_id: str, **_: Any) -> dict[str, Any]:
        self._record("restore", {"server_id": server_id})
        return self._come_up(next(s["mode"] for s in self.servers if s["id"] == server_id))

    def _start(self, mode: str, **_: Any) -> dict[str, Any]:
        if not self._record("start", {"mode": mode}):
            return {"success": False, "error": "port in use"}
        return self._come_up(mode)

    def list(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self.servers]


def enabled_remote() -> mock.Mock:
    remote = mock.Mock(valid=True, enabled=True, routable=True, api_key_env="", key_required=False,
                       provider="llamacpp", base_url="http://thanatos:8080", model="qwen3-32b", aliases=[],
                       display_name="Qwen on Thanatos", summary="Thanatos · Self-hosted")
    remote.name = "thanatos"
    return remote


class WithoutSwitchingTests(unittest.TestCase):
    def test_a_stopped_profile_is_never_started(self) -> None:
        with self.assertRaises(GatewayError) as ctx:
            Router(endpoints=lambda: [], servers=lambda: []).resolve("llama")
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(Router(endpoints=lambda: [], servers=lambda: []).loadable(), [])


class SwitcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeControlApi([server("qwen")])
        self.api.install(self)
        self.switcher = ModelSwitcher(self.api, drain_timeout=2)

    def router(self, endpoints=()):
        return Router(endpoints=lambda: list(endpoints), servers=self.api.list, switcher=self.switcher)

    def resolve(self, model, endpoints=()):
        return self.router(endpoints).resolve(model)

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

        self.api.servers[:] = [server("qwen"), server("llama", suspended=True)]
        self.api.calls.clear()
        self.resolve("llama")
        self.assertEqual(self.api.calls, [("release", {"server_id": "qwen-id"}), ("resume", {"server_id": "llama-id"})])

    def test_unlaunchable_or_unknown_model_does_not_switch(self) -> None:
        self.assertEqual(self.resolve("broken").model_id, "qwen")
        self.assertEqual(self.resolve("gpt-4o").model_id, "qwen")
        self.assertEqual(self.api.calls, [])

    def test_failures_are_503s_that_name_the_step(self) -> None:
        self.api.fail = "start"
        with self.assertRaises(GatewayError) as ctx:
            self.resolve("llama")
        self.assertEqual(ctx.exception.status, 503)
        self.assertIn("could not switch to llama: port in use", str(ctx.exception))

        self.api.fail = "profiles"
        self.switcher = ModelSwitcher(self.api)
        with self.assertRaises(GatewayError) as ctx:
            self.resolve("llama")
        self.assertIn("needs inferencedeck-web", str(ctx.exception))

    def test_switch_waits_for_in_flight_requests(self) -> None:
        in_flight = {"qwen-id": 1}
        self.api.busy = lambda: dict(in_flight)
        released_at: list[float] = []
        original = self.api._release

        def release(server_id):
            released_at.append(time.monotonic())
            return original(server_id)

        self.api._release = release
        self.api.install(self)
        finished_at: list[float] = []

        def finish():
            time.sleep(0.3)
            finished_at.append(time.monotonic())
            in_flight.clear()

        threading.Thread(target=finish).start()
        self.resolve("llama")
        self.assertTrue(released_at and finished_at)
        self.assertGreaterEqual(released_at[0], finished_at[0])

    def test_switch_gives_up_instead_of_cutting_off_a_request(self) -> None:
        self.api.busy = lambda: {"qwen-id": 1}
        self.switcher = ModelSwitcher(self.api, drain_timeout=0.3)
        with self.assertRaises(GatewayError) as ctx:
            self.resolve("llama")
        self.assertEqual(ctx.exception.status, 503)
        self.assertIn("still serving qwen (1 request)", str(ctx.exception))
        self.assertEqual(self.api.calls, [])  # qwen kept running, llama not started
        self.assertTrue(self.api.servers[0]["running"])

    def test_switch_waits_for_a_request_another_gateway_process_is_serving(self) -> None:
        # The gateway that asks for the switch has nothing running; another one
        # (its own in-flight file) is streaming from qwen.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(inflight, "inflight_dir", return_value=Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        other_gateway = inflight.Tracker(path=Path(tmp.name) / "gateway-other.json", pid=os.getpid())
        other_gateway.begin("qwen-id")
        self.api.busy = None  # read every gateway's in-flight file, as inferencedeck-web does
        self.switcher = ModelSwitcher(self.api, drain_timeout=0.3)
        with self.assertRaises(GatewayError):
            self.resolve("llama")
        self.assertEqual(self.api.calls, [])

        threading.Timer(0.2, other_gateway.end, args=("qwen-id",)).start()
        self.switcher = ModelSwitcher(self.api, drain_timeout=5)
        self.assertEqual(self.resolve("llama").model_id, "llama")
        self.assertEqual(self.api.calls, [("release", {"server_id": "qwen-id"}), ("start", {"mode": "llama"})])

    def test_request_starting_just_before_the_release_is_waited_for(self) -> None:
        # Idle at the first look, busy at the check right before the release, then idle.
        looks = iter([{}, {"qwen-id": 1}, {"qwen-id": 1}])
        self.api.busy = lambda: next(looks, {})
        self.resolve("llama")
        self.assertEqual(self.api.calls, [("release", {"server_id": "qwen-id"}), ("start", {"mode": "llama"})])
        self.assertIsNone(next(looks, None))  # it looked again before releasing

    def test_concurrent_requests_switch_once(self) -> None:
        results: list[str] = []
        threads = [threading.Thread(target=lambda: results.append(self.resolve("llama").model_id)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, ["llama"] * 4)
        self.assertEqual([c[0] for c in self.api.calls].count("start"), 1)

    def test_concurrent_switches_from_separate_gateways_switch_once(self) -> None:
        # Separate ModelSwitchers share nothing in-process; model_switch serializes them.
        results: list[str] = []

        def request() -> None:
            results.append(Router(endpoints=lambda: [], servers=self.api.list,
                                  switcher=ModelSwitcher(self.api)).resolve("llama").model_id)

        threads = [threading.Thread(target=request) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, ["llama"] * 4)
        self.assertEqual([c[0] for c in self.api.calls], ["release", "start"])

    def test_running_profile_matched_by_name_or_alias_needs_no_switch(self) -> None:
        # The catalog knows a server by mode/id/file; name and alias come from the profile.
        target = self.resolve("qwen3-32b")
        self.assertEqual(target.model_id, "qwen")
        self.assertEqual(self.api.calls, [])

    def test_no_switch_while_a_remote_endpoint_is_enabled(self) -> None:
        target = self.resolve("llama", endpoints=[enabled_remote()])
        self.assertEqual(target.endpoint, "thanatos")
        self.assertEqual(self.api.calls, [])

    def test_loadable_lists_profiles_that_are_not_running(self) -> None:
        self.assertEqual([p["mode"] for p in self.router().loadable()], ["llama"])


class ExplainSwitchTests(unittest.TestCase):
    """``/route`` says what a request would do, switch included, without doing it."""

    def setUp(self) -> None:
        self.api = FakeControlApi([server("qwen")])
        self.api.install(self)
        self.router = Router(endpoints=lambda: [], servers=self.api.list, switcher=ModelSwitcher(self.api))

    def test_a_request_that_would_switch_reports_the_switch(self) -> None:
        explained = self.router.explain("llama-3.3-70b")
        self.assertEqual(explained["action"], "switch")
        self.assertEqual(explained["would_load"], "llama")
        self.assertEqual(explained["would_release"], ["qwen"])
        self.assertEqual(explained["target"]["model"], "llama")
        self.assertFalse(explained["target"]["loaded"])
        self.assertEqual(self.api.calls, [])
        # ...and the request itself goes where explain said.
        self.assertEqual(self.router.resolve("llama-3.3-70b").model_id, explained["target"]["model"])

    def test_a_loaded_model_routes_without_a_switch(self) -> None:
        explained = self.router.explain("qwen")
        self.assertEqual(explained["action"], "route")
        self.assertEqual(explained["target"]["model"], "qwen")
        self.assertTrue(explained["target"]["loaded"])
        self.assertNotIn("would_load", explained)

    def test_nothing_running_still_explains_a_load(self) -> None:
        self.api.servers.clear()
        explained = self.router.explain("llama")
        self.assertEqual((explained["action"], explained["would_load"], explained["would_release"]),
                         ("switch", "llama", []))
        self.assertEqual(self.api.calls, [])

    def test_without_switching_nothing_is_predicted_to_load(self) -> None:
        router = Router(endpoints=lambda: [], servers=self.api.list)
        explained = router.explain("llama")
        self.assertEqual((explained["action"], explained["target"]["model"]), ("route", "qwen"))


class SwitchingGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeUpstream.received, _FakeUpstream.headers_seen, _FakeUpstream.reply_tool = [], [], False
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        port = upstream.server_address[1]
        self.api = FakeControlApi([server("qwen", port)], port=port)
        self.api.install(self)
        counts = tempfile.TemporaryDirectory()
        self.addCleanup(counts.cleanup)
        self.inflight = inflight.Tracker(path=Path(counts.name) / "gateway.json")
        self.switcher = ModelSwitcher(self.api)
        router = Router(endpoints=lambda: [], servers=self.api.list, switcher=self.switcher)
        gateway = make_server("127.0.0.1", 0, AuthState(token=""), router=router, inflight=self.inflight)
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

    def test_streamed_request_is_counted_until_it_ends(self) -> None:
        status, raw = self.post({"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertIn(b"[DONE]", raw)
        counted = self.inflight.counts()["qwen-id"]
        self.assertEqual((counted["in_flight"], counted["requests"]), (0, 1))

    def test_route_reports_the_switch_a_request_would_make(self) -> None:
        with urllib.request.urlopen(self.base + "/route?model=llama", timeout=10) as response:
            body = json.loads(response.read())
        self.assertEqual((body["action"], body["would_load"]), ("switch", "llama"))
        self.assertEqual(self.api.calls, [])

    def test_busy_switch_is_a_503(self) -> None:
        self.api.busy = lambda: {"qwen-id": 1}
        self.switcher.drain_timeout = 0.2
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post({"model": "llama", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(ctx.exception.code, 503)
        self.assertIn("still serving qwen", json.loads(ctx.exception.read())["error"]["message"])

    def test_models_endpoint(self) -> None:
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as response:
            data = json.loads(response.read())["data"]
        self.assertEqual([(m["id"], m["loaded"], m["default"]) for m in data],
                         [("qwen", True, True), ("llama", False, False)])


if __name__ == "__main__":
    unittest.main()
