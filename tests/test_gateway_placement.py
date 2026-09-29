from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from inferencedeck import fleet
from inferencedeck.auth import AuthState
from inferencedeck.gateway import placement, router
from inferencedeck.gateway.server import make_server
from inferencedeck.remotes import list_endpoints
from tests.test_gateway import GatewayServerTests, _FakeUpstream, _write_endpoints


def _host(name: str, *, local: bool = False, url: str = "", servers=(), benchmarks=(), reachable: bool = True) -> dict[str, Any]:
    return {"name": name, "local": local, "url": url, "reachable": reachable, "error": None,
            "servers": list(servers), "benchmarks": list(benchmarks), "free_vram_bytes": None}


def _qwen(tps: float | None = None, suspended: bool = False) -> dict[str, Any]:
    return {"profile": "qwen", "running": True, "suspended": suspended, "tokens_per_second": tps}


class _Fleet:
    def __init__(self, overview):
        self.overview = overview

    def overview_nowait(self):
        return self.overview


def _ranker(overview, peers=True, enabled=True) -> placement.FleetRanker:
    config = SimpleNamespace(fleet_peers=[{"url": "http://friday:8716"}] if peers else [], fleet_name="", gateway_placement=enabled)
    return placement.FleetRanker(config=lambda: config, get_fleet=lambda *_: _Fleet(overview))


class RankHostsTests(unittest.TestCase):
    def test_groups_and_host_keys(self) -> None:
        overview = {"hosts": [
            _host("thanatos", local=True, servers=[_qwen(20)]),
            _host("friday", url="http://friday.tail.ts.net:8716", servers=[_qwen(suspended=True)]),
            _host("p70", url="http://10.0.0.7:8716"),
        ]}
        ranks = placement.rank_hosts(overview, "qwen")
        self.assertEqual(ranks[placement.LOCAL].group, placement.LOADED)
        self.assertEqual(ranks["friday"].group, placement.PAUSED)
        self.assertEqual(ranks["friday.tail.ts.net"], ranks["friday"])
        self.assertEqual(ranks["10.0.0.7"].group, placement.AVAILABLE)
        self.assertEqual(placement.rank_target(("elsewhere",), ranks).group, placement.UNKNOWN)

    def test_a_peer_sharing_the_local_name_keeps_its_own_identity(self) -> None:
        # Config validation rejects this now, but placement must not depend on it:
        # the remote's rank is keyed by its own name/URL, never by @local.
        overview = {"hosts": [
            _host("thanatos", local=True, servers=[]),
            _host("thanatos", url="http://other-box:8716", servers=[_qwen(30)]),
        ]}
        ranks = placement.rank_hosts(overview, "qwen")
        self.assertEqual(ranks["other-box"].group, placement.LOADED)
        self.assertEqual(ranks[placement.LOCAL].group, placement.AVAILABLE)

    def test_ranker_is_off_without_peers_or_when_disabled_or_broken(self) -> None:
        overview = {"hosts": [_host("thanatos", local=True, servers=[_qwen()])]}
        self.assertIsNone(_ranker(overview, peers=False).ranks("qwen"))
        self.assertIsNone(_ranker(overview, enabled=False).ranks("qwen"))
        self.assertIsNone(_ranker(None).ranks("qwen"))  # no fleet view yet
        broken = placement.FleetRanker(config=mock.Mock(side_effect=OSError("unreadable")))
        self.assertIsNone(broken.ranks("qwen"))


class OverviewNowaitTests(unittest.TestCase):
    def test_first_call_never_waits_then_serves_the_cache(self) -> None:
        release = threading.Event()

        def slow_collect():
            release.wait(5)
            return {"gpus": [], "servers": []}

        view = fleet.Fleet([], "me", collect_local=slow_collect, max_age=60)
        started = time.monotonic()
        self.assertIsNone(view.overview_nowait())
        self.assertLess(time.monotonic() - started, 1)
        release.set()
        deadline = time.monotonic() + 5
        while view.overview_nowait() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(view.overview_nowait()["hosts"][0]["name"], "me")


class RouterPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _write_endpoints(self.root, friday={"provider": "llamacpp", "model": "qwen", "baseUrl": "http://friday.tail.ts.net:8080/v1"})
        self.servers = [{"id": "qwen-1", "mode": "qwen", "running": True, "host": "127.0.0.1", "port": 8080}]

    def _router(self, ranker) -> router.Router:
        return router.Router(endpoints=lambda: list_endpoints(self.root), servers=lambda: self.servers, ranker=ranker)

    def test_catalog_order_without_fleet(self) -> None:
        target = self._router(None).resolve("qwen")
        self.assertEqual(target.server_id, "qwen-1")  # locals come before other endpoints

    def test_loaded_elsewhere_beats_not_loaded_here(self) -> None:
        overview = {"hosts": [
            _host("thanatos", local=True, servers=[]),
            _host("friday", url="http://friday.tail.ts.net:8716", servers=[_qwen(31.8)]),
        ]}
        target = self._router(_ranker(overview)).resolve("qwen")
        self.assertEqual(target.endpoint, "friday")

    def test_faster_machine_wins_when_both_have_it_loaded(self) -> None:
        overview = {"hosts": [
            _host("thanatos", local=True, servers=[_qwen(12.0)]),
            _host("friday", url="http://friday.tail.ts.net:8716", servers=[_qwen(31.8)]),
        ]}
        self.assertEqual(self._router(_ranker(overview)).resolve("qwen").endpoint, "friday")
        overview["hosts"][1]["servers"] = [_qwen(8.0)]
        self.assertEqual(self._router(_ranker(overview)).resolve("qwen").server_id, "qwen-1")

    def test_paused_machine_goes_last_and_unknown_machines_keep_their_place(self) -> None:
        overview = {"hosts": [
            _host("thanatos", local=True, servers=[_qwen(12.0)]),
            _host("friday", url="http://friday.tail.ts.net:8716", servers=[_qwen(suspended=True)]),
        ]}
        self.assertEqual(self._router(_ranker(overview)).resolve("qwen").server_id, "qwen-1")

    def test_explain_target_is_its_first_candidate(self) -> None:
        # The fleet view arrives between two lookups: the first sees none, the second sees friday loaded.
        views = iter([None, {"hosts": [
            _host("thanatos", local=True, servers=[]),
            _host("friday", url="http://friday.tail.ts.net:8716", servers=[_qwen(31.8)]),
        ]}])
        config = SimpleNamespace(fleet_peers=[{"url": "http://friday:8716"}], fleet_name="", gateway_placement=True)
        ranker = placement.FleetRanker(config=lambda: config, get_fleet=lambda *_: _Fleet(next(views, None)))
        report = self._router(ranker).explain("qwen")
        first = report["candidates"][0]
        self.assertEqual({k: report["target"][k] for k in report["target"] if k in first},
                         {k: first[k] for k in report["target"] if k in first})

    def test_single_match_never_consults_the_fleet(self) -> None:
        ranker = mock.Mock()
        self.servers = []
        self.assertEqual(self._router(ranker).resolve("qwen").endpoint, "friday")
        ranker.ranks.assert_not_called()

    def test_explain_reports_the_ranking_without_loading_anything(self) -> None:
        overview = {"hosts": [
            _host("thanatos", local=True, servers=[]),
            _host("friday", url="http://friday.tail.ts.net:8716", servers=[_qwen(31.8)]),
        ]}
        switcher = mock.Mock()
        switcher.find_profile.return_value = None
        r = router.Router(endpoints=lambda: list_endpoints(self.root), servers=lambda: self.servers,
                          ranker=_ranker(overview), switcher=switcher)
        explained = r.explain("qwen")
        self.assertTrue(explained["placement_used"])
        self.assertEqual([c["group"] for c in explained["candidates"]], ["loaded here", "not loaded here"])
        self.assertEqual(explained["candidates"][0]["fleet_host"], "friday")
        r.explain("never-heard-of-it")
        switcher.ensure_loaded.assert_not_called()


class GatewayPlacementServerTests(unittest.TestCase):
    """Two upstreams answer to the same name; the fleet view picks, and the reply says which."""

    def _upstream(self) -> int:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def setUp(self) -> None:
        _FakeUpstream.received, _FakeUpstream.headers_seen, _FakeUpstream.reply_tool = [], [], False
        self.local_port, self.remote_port = self._upstream(), self._upstream()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        _write_endpoints(root, friday={"provider": "llamacpp", "model": "qwen", "host": "friday",
                                       "baseUrl": f"http://127.0.0.1:{self.remote_port}/v1"})
        servers = [{"id": "qwen-1", "mode": "qwen", "running": True, "host": "127.0.0.1", "port": self.local_port}]
        overview = {"hosts": [_host("thanatos", local=True, servers=[_qwen(10.0)]),
                              _host("friday", url="http://friday:8716", servers=[_qwen(40.0)])]}
        r = router.Router(endpoints=lambda: list_endpoints(root), servers=lambda: servers, ranker=_ranker(overview))
        gateway = make_server("127.0.0.1", 0, AuthState(token=""), r)
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        self.base = f"http://127.0.0.1:{gateway.server_address[1]}"

    post = GatewayServerTests.post

    def test_request_goes_to_the_faster_machine_and_says_so(self) -> None:
        request = urllib.request.Request(self.base + "/v1/chat/completions", method="POST",
                                         data=json.dumps({"model": "qwen", "messages": [{"role": "user", "content": "x"}]}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            self.assertEqual(response.headers["X-InferenceDeck-Target"], "qwen @ friday")
        self.assertTrue(_FakeUpstream.headers_seen[-1]["Host"].endswith(f":{self.remote_port}"))

    def test_route_endpoint_explains(self) -> None:
        with urllib.request.urlopen(self.base + "/route?model=qwen", timeout=10) as response:
            body = json.loads(response.read())
        self.assertEqual([c["fleet_host"] for c in body["candidates"]], ["friday", "thanatos"])


if __name__ == "__main__":
    unittest.main()
