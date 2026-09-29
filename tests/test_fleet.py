from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from inferencedeck import fleet
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler

GB = 1024**3


def _snap(servers=(), gpus=None, benchmarks=()) -> dict:
    return {
        "timestamp": "2026-09-27T10:00:00+00:00",
        "system": {"cpu_percent": 10.0, "memory": {"total_bytes": 32 * GB}},
        "gpus": gpus if gpus is not None else [{"index": 0, "name": "RTX 3090", "utilization_percent": 50.0, "vram_used_bytes": 4 * GB, "vram_total_bytes": 24 * GB}],
        "servers": list(servers),
        "benchmarks": list(benchmarks),
    }


class _PeerHandler(BaseHTTPRequestHandler):
    seen_auth: list = []
    mode = "ok"

    def log_message(self, *args) -> None:
        return

    def do_GET(self) -> None:
        type(self).seen_auth.append(self.headers.get("Authorization"))
        if self.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
            self.end_headers()
            return
        if self.mode == "401":
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps(_snap() if self.mode == "ok" else ["not", "a", "snapshot"]).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class PeerConfigTests(unittest.TestCase):
    def test_parse_peers_validates_and_dedupes(self) -> None:
        peers, errors = fleet.parse_peers([
            {"name": "thanatos", "url": "https://thanatos.ts.net:8716/", "tokenEnv": "THANATOS_TOKEN"},
            {"url": "http://friday:8716"},
            {"name": "Thanatos", "url": "http://other:8716"},
            {"name": "bad", "url": "friday:8716"},
            "nope",
        ])
        self.assertEqual([(p.name, p.url, p.token_env) for p in peers], [
            ("thanatos", "https://thanatos.ts.net:8716", "THANATOS_TOKEN"),
            ("friday", "http://friday:8716", ""),
        ])
        self.assertEqual(len(errors), 3)

    def test_peer_description_never_includes_the_token(self) -> None:
        peer = fleet.Peer("thanatos", "http://t:8716", "THANATOS_TOKEN")
        with mock.patch.dict(os.environ, {"THANATOS_TOKEN": "s3cret"}):
            described = json.dumps(peer.to_dict())
        self.assertNotIn("s3cret", described)
        self.assertIn('"token_present": true', described)


class FetchTests(unittest.TestCase):
    def _serve(self, mode: str = "ok") -> str:
        handler = type("P", (_PeerHandler,), {"seen_auth": [], "mode": mode})
        self.handler = handler
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        return f"http://127.0.0.1:{server.server_port}"

    def test_reads_snapshot_with_bearer_token(self) -> None:
        peer = fleet.Peer("friday", self._serve(), "FRIDAY_TOKEN")
        with mock.patch.dict(os.environ, {"FRIDAY_TOKEN": "abc"}):
            result = fleet.fetch_peer(peer)
        self.assertTrue(result["reachable"])
        self.assertEqual(result["snapshot"]["gpus"][0]["name"], "RTX 3090")
        self.assertEqual(self.handler.seen_auth, ["Bearer abc"])
        self.assertIsNotNone(result["latency_ms"])

    def test_no_token_env_sends_no_authorization(self) -> None:
        fleet.fetch_peer(fleet.Peer("friday", self._serve()))
        self.assertEqual(self.handler.seen_auth, [None])

    def test_unauthorized_redirect_bad_body_and_dead_peer(self) -> None:
        self.assertEqual(fleet.fetch_peer(fleet.Peer("a", self._serve("401")))["error"], "HTTP 401 (check its token)")
        redirected = fleet.fetch_peer(fleet.Peer("b", self._serve("redirect"), "T"))
        self.assertFalse(redirected["reachable"])
        self.assertIn("302", redirected["error"])
        self.assertIn("not a telemetry snapshot", fleet.fetch_peer(fleet.Peer("c", self._serve("list")))["error"])
        self.assertFalse(fleet.fetch_peer(fleet.Peer("d", "http://127.0.0.1:9"), timeout=0.5)["reachable"])


class OverviewTests(unittest.TestCase):
    def test_local_and_peer_rows_with_failures_isolated(self) -> None:
        peers = [fleet.Peer("friday", "http://f:1"), fleet.Peer("p70", "http://p:1")]

        def fetch(peer):
            if peer.name == "p70":
                raise RuntimeError("boom")
            return {"reachable": True, "latency_ms": 12.0, "snapshot": _snap(servers=[{"profile": "llama-8b", "running": True, "tokens_per_second": 76.2}])}

        view = fleet.Fleet(peers, "thanatos", collect_local=lambda: _snap(), fetch=fetch).overview()
        names = [(h["name"], h["local"], h["reachable"]) for h in view["hosts"]]
        self.assertEqual(names, [("thanatos", True, True), ("friday", False, True), ("p70", False, False)])
        self.assertEqual(view["hosts"][0]["free_vram_bytes"], 20 * GB)
        self.assertEqual(view["hosts"][1]["servers"][0]["tokens_per_second"], 76.2)
        self.assertEqual(view["hosts"][2]["error"], "boom")

    def test_malformed_peer_snapshot_marks_only_that_peer(self) -> None:
        peers = [fleet.Peer("friday", "http://f:1"), fleet.Peer("junk", "http://j:1")]

        def fetch(peer):
            if peer.name == "junk":  # reachable, but not InferenceDeck telemetry
                return {"reachable": True, "latency_ms": 3.0, "snapshot": {"gpus": ["x"], "servers": "nope"}}
            return {"reachable": True, "latency_ms": 12.0, "snapshot": _snap()}

        view = fleet.Fleet(peers, "thanatos", collect_local=lambda: _snap(), fetch=fetch).overview()
        names = [(h["name"], h["reachable"]) for h in view["hosts"]]
        self.assertEqual(names, [("thanatos", True), ("friday", True), ("junk", False)])
        self.assertIn("unreadable telemetry", view["hosts"][2]["error"])

    def test_overview_is_cached(self) -> None:
        collect = mock.Mock(return_value=_snap())
        view = fleet.Fleet([], "x", collect_local=collect, max_age=60)
        view.overview()
        view.overview()
        collect.assert_called_once()

    def test_current_rebuilds_when_config_changes(self) -> None:
        with mock.patch.object(fleet, "_fleet", None):
            first = fleet.current([{"url": "http://a:1"}], "me")
            self.assertIs(fleet.current([{"url": "http://a:1"}], "me"), first)
            self.assertIsNot(fleet.current([{"url": "http://b:1"}], "me"), first)


class PlacementTests(unittest.TestCase):
    def _host(self, name, servers=(), benchmarks=(), free=10, reachable=True, local=False):
        return {"name": name, "local": local, "reachable": reachable, "error": None if reachable else "HTTP 401",
                "servers": list(servers), "benchmarks": list(benchmarks), "free_vram_bytes": free * GB}

    def test_loaded_beats_paused_beats_known_beats_unseen(self) -> None:
        view = {"hosts": [
            self._host("unseen", free=40),
            self._host("known", benchmarks=[{"profile": "qwen", "tokens_per_second": 40}]),
            self._host("paused", servers=[{"profile": "qwen", "running": True, "suspended": True}]),
            self._host("loaded", servers=[{"profile": "qwen", "running": True, "tokens_per_second": 22.0, "requests_active": 1}]),
            self._host("down", reachable=False),
        ]}
        result = fleet.placement(view, profile="Qwen")
        self.assertEqual([c["host"] for c in result["candidates"]], ["loaded", "paused", "known", "unseen"])
        self.assertEqual(result["recommended"], "loaded")
        self.assertEqual(result["unreachable"], [{"host": "down", "error": "HTTP 401"}])
        self.assertIn("22 tok/s now", result["candidates"][0]["reasons"])
        self.assertIn("1 request in progress", result["candidates"][0]["reasons"])

    def test_within_a_tier_faster_then_less_busy_then_more_vram(self) -> None:
        view = {"hosts": [
            self._host("slow", benchmarks=[{"profile": "qwen", "tokens_per_second": 20}]),
            self._host("fast", benchmarks=[{"profile": "qwen", "tokens_per_second": 31}]),
        ]}
        self.assertEqual(fleet.placement(view, profile="qwen")["recommended"], "fast")

    def test_matches_on_model_file_and_reports_nothing_when_unseen(self) -> None:
        view = {"hosts": [self._host("a", servers=[{"profile": "big", "model": "Qwen3-30B-Q4_K_M.gguf", "running": True}]), self._host("b")]}
        self.assertEqual(fleet.placement(view, model="qwen3-30b-q4_k_m.gguf")["recommended"], "a")
        self.assertIsNone(fleet.placement({"hosts": [self._host("b")]}, profile="qwen")["recommended"])
        with self.assertRaises(ValueError):
            fleet.placement(view)


class FleetApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = ControlPlane()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        self.base = f"http://127.0.0.1:{server.server_port}"
        view = fleet.Fleet([], "thanatos", collect_local=lambda: _snap(servers=[{"profile": "qwen", "running": True}]))
        patcher = mock.patch.object(fleet, "current", return_value=view)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fleet_and_placement_endpoints(self) -> None:
        with urllib.request.urlopen(self.base + "/api/fleet", timeout=5) as response:
            self.assertEqual(json.loads(response.read())["hosts"][0]["name"], "thanatos")
        with urllib.request.urlopen(self.base + "/api/fleet/placement?profile=qwen", timeout=5) as response:
            self.assertEqual(json.loads(response.read())["recommended"], "thanatos")
        with self.assertRaises(urllib.error.HTTPError) as bad:
            urllib.request.urlopen(self.base + "/api/fleet/placement", timeout=5)
        self.assertEqual(bad.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
