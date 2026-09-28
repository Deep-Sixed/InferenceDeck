"""Update checks in the control API, the web UI and the Windows/macOS tray."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import control, tray
from inferencedeck.auth import AuthState
from inferencedeck.config import AppConfig
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler
from inferencedeck.schema import Environment

HAS_PYSTRAY = importlib.util.find_spec("pystray") is not None and importlib.util.find_spec("PIL") is not None
RUNNING = {"id": "qwen-1", "mode": "qwen", "running": True, "status": "running", "port": 8080, "ctx_size": 32768}

WEB = Path(__file__).resolve().parents[1] / "inferencedeck" / "web"
LINUX_TRAY = Path(__file__).resolve().parents[1] / "frontends" / "linux" / "inferencedeck_tray.py"

VLLM_CPP_UPDATE = {
    "runtime_id": "vllm.cpp",
    "runtime_name": "vllm.cpp",
    "current_version": "0.0.2",
    "latest_version": "v0.0.3",
    "update_available": True,
    "channel": "stable",
    "release_url": "https://github.com/mudler/vllm.cpp/releases/tag/v0.0.3",
}
MLC_CURRENT = {
    "runtime_id": "mlc-llm",
    "runtime_name": "MLC LLM",
    "current_version": "0.20.0",
    "latest_version": "v0.20.0",
    "update_available": False,
    "channel": "stable",
    "release_url": "https://github.com/mlc-ai/mlc-llm/releases/tag/v0.20.0",
}
PAYLOAD = {"channel": "stable", "updates": [VLLM_CPP_UPDATE, MLC_CURRENT], "skipped_no_version": ["ollama"]}


class ControlPlaneUpdatesTests(unittest.TestCase):
    def test_checks_detected_runtimes_on_the_configured_channel(self) -> None:
        detected = [Environment(id="vllm.cpp", kind="local_binary", name="vllm.cpp", available=True, version="0.0.2")]
        with mock.patch.object(control, "detect_all", return_value=detected), mock.patch.object(
            control, "check_runtime_updates", return_value=PAYLOAD
        ) as check, mock.patch.object(ControlPlane, "_config", return_value=AppConfig(update_channel="prerelease")):
            result = ControlPlane(project_root=".").updates(refresh=True)
        self.assertIs(result, PAYLOAD)
        self.assertEqual(check.call_args.args[0][0]["id"], "vllm.cpp")
        self.assertEqual(check.call_args.kwargs, {"channel": "prerelease", "force_refresh": True})


class UpdatesApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.updates.return_value = PAYLOAD
        handler.auth_state = AuthState(username="admin", token="s3cret")
        self.control = handler.control_plane
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.url + path, headers={"X-Auth-Token": "s3cret"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)

    def test_endpoint_serves_cached_results_and_refreshes_on_request(self) -> None:
        self.assertEqual(self._get("/api/updates")["updates"][0]["runtime_id"], "vllm.cpp")
        self.control.updates.assert_called_with(refresh=False)
        self._get("/api/updates?refresh=1")
        self.control.updates.assert_called_with(refresh=True)

    def test_endpoint_requires_auth(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.url + "/api/updates", timeout=5)
        self.assertEqual(ctx.exception.code, 401)
        self.control.updates.assert_not_called()

    def test_tray_client_reads_the_endpoint(self) -> None:
        api = tray.ApiClient(self.url, token="s3cret")
        self.assertEqual(api.updates(refresh=True)["channel"], "stable")
        self.control.updates.assert_called_with(refresh=True)


class TrayUpdateHelpersTests(unittest.TestCase):
    def test_labels(self) -> None:
        self.assertEqual(tray.updates_label(None), "Runtime updates: checking…")
        self.assertEqual(tray.updates_label(PAYLOAD), "Runtime updates: 1 available")
        self.assertEqual(tray.updates_label({"updates": [MLC_CURRENT]}), "Runtime updates: up to date")
        self.assertEqual(tray.updates_label({"updates": []}), "Runtime updates: nothing to check")
        self.assertEqual(tray.updates_label(PAYLOAD, error="offline"), "Runtime updates: check failed")
        self.assertEqual(tray.update_entry_label(VLLM_CPP_UPDATE), "vllm.cpp 0.0.2 → v0.0.3")

    def test_only_github_release_pages_are_opened(self) -> None:
        self.assertEqual(tray.release_url(VLLM_CPP_UPDATE), VLLM_CPP_UPDATE["release_url"])
        for url in ("http://github.com/x", "https://evil.example/github.com/", "javascript:alert(1)", None):
            self.assertIsNone(tray.release_url({**VLLM_CPP_UPDATE, "release_url": url}), url)


class TrayControllerUpdateTests(unittest.TestCase):
    def _controller(self):
        api = mock.Mock(spec=tray.ApiClient)
        api.updates.return_value = PAYLOAD
        notes: list[str] = []
        changes: list[bool] = []
        c = tray.TrayController(api, notes.append, lambda: changes.append(True), background=lambda fn: fn())
        c.state = tray.TrayState(servers=[RUNNING])
        return c, api, notes, changes

    def test_background_check_stores_results_quietly(self) -> None:
        c, api, notes, changes = self._controller()
        c.check_updates()
        api.updates.assert_called_once_with(refresh=False)
        self.assertIs(c.updates, PAYLOAD)
        self.assertEqual(notes, [])
        self.assertEqual(changes, [True])

    def test_check_now_reports_the_outcome(self) -> None:
        c, api, notes, _ = self._controller()
        c.check_updates(refresh=True)
        api.updates.assert_called_once_with(refresh=True)
        self.assertEqual(notes, ["1 runtime update(s) available."])
        api.updates.side_effect = tray.ApiError("InferenceDeck is not reachable")
        c.check_updates(refresh=True)
        self.assertIn("not reachable", c.updates_error)
        self.assertIn("Update check failed", notes[-1])

    def test_checks_run_at_startup_then_hourly_and_not_while_offline(self) -> None:
        c, _api, _notes, _ = self._controller()
        self.assertTrue(c.updates_due(now=100.0))
        c.check_updates()
        started = c._last_update_check
        self.assertFalse(c.updates_due(now=started + tray.UPDATE_CHECK_SECONDS - 1))
        self.assertTrue(c.updates_due(now=started + tray.UPDATE_CHECK_SECONDS))
        c.state = tray.TrayState(error="unreachable")
        self.assertFalse(c.updates_due(now=started + 10 * tray.UPDATE_CHECK_SECONDS))

    def test_due_exactly_an_interval_later_whatever_the_clock_reads(self) -> None:
        # time.monotonic() is uptime; for some values (t + 3600) - t rounds to
        # just under 3600, which made the check above fail on some CI runners.
        c, _api, _notes, _ = self._controller()
        for last in (523312.47000279423, 4194145.3468878143, 100.0):
            c._last_update_check = last
            self.assertTrue(c.updates_due(now=last + tray.UPDATE_CHECK_SECONDS), last)
            self.assertFalse(c.updates_due(now=last + tray.UPDATE_CHECK_SECONDS - 1), last)


@unittest.skipUnless(HAS_PYSTRAY, "pip install inferencedeck[tray]")
class PystrayUpdateMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["PYSTRAY_BACKEND"] = "dummy"
        import pystray

        cls.pystray = pystray

    def test_menu_lists_updates_and_opens_release_pages(self) -> None:
        api = mock.Mock(spec=tray.ApiClient)
        api.updates.return_value = PAYLOAD
        c = tray.TrayController(api, lambda _m: None, background=lambda fn: fn())
        c.state = tray.TrayState(servers=[RUNNING])
        menu = tray.build_menu(self.pystray, c, lambda: None, lambda: None)
        icon = self.pystray.Icon("t", tray.make_icon_image("idle"), "t", menu)
        labels = lambda: {str(i.text): i for i in menu.items}  # noqa: E731
        self.assertIn("Runtime updates: checking…", labels())
        c.updates = PAYLOAD
        item = labels()["Runtime updates: 1 available"]
        entries = {str(i.text): i for i in item.submenu.items}
        self.assertNotIn("MLC LLM 0.20.0 → v0.20.0", entries)  # up to date: not listed
        with mock.patch.object(tray.webbrowser, "open") as open_url:
            entries[tray.menu_text("vllm.cpp 0.0.2 → v0.0.3")](icon)
        open_url.assert_called_once_with(VLLM_CPP_UPDATE["release_url"])
        entries["Check now"](icon)
        api.updates.assert_called_with(refresh=True)


class StaticFrontendTests(unittest.TestCase):
    def test_web_ui_has_the_updates_card(self) -> None:
        html = (WEB / "index.html").read_text(encoding="utf-8")
        script = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="updates"', html)
        self.assertIn('id="updates-check"', html)
        self.assertIn("/api/updates", script)
        # Release links are limited to GitHub and open without an opener reference.
        self.assertIn("^https:\\/\\/github\\.com\\/", script)
        self.assertIn('rel="noopener noreferrer"', script)

    def test_linux_tray_checks_updates_hourly_with_a_long_timeout(self) -> None:
        source = LINUX_TRAY.read_text(encoding="utf-8")
        self.assertIn("UPDATE_CHECK_SECONDS = 3600", source)
        self.assertIn('"/api/updates"', source)
        self.assertIn('url.startswith("https://github.com/")', source)


if __name__ == "__main__":
    unittest.main()
