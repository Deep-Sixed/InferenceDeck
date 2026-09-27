from __future__ import annotations

import importlib.util
import os
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

from inferencedeck import tray
from inferencedeck.auth import AuthState
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler

HAS_PYSTRAY = importlib.util.find_spec("pystray") is not None and importlib.util.find_spec("PIL") is not None

RUNNING = {"id": "qwen-1", "mode": "qwen", "running": True, "status": "running", "port": 8080, "ctx_size": 32768,
           "model_path": "/m/qwen.gguf", "command_line": "llama-server -m qwen.gguf"}
PAUSED = {**RUNNING, "suspended": True, "status": "suspended"}
PARKED = {"id": "qwen-1", "mode": "qwen", "running": False, "status": "parked", "pid": None, "ctx_size": 8192}


class ApiClientTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.status.return_value = {"servers": [RUNNING], "remote_active": None}
        handler.control_plane.profiles.return_value = [{"mode": "qwen", "launchable": True}]
        handler.control_plane.remote_endpoints.return_value = {"endpoints": [{"name": "oai"}]}
        handler.control_plane.start.return_value = {"success": False, "error": "Profile 'qwen' already has a tracked running server."}
        handler.auth_state = AuthState(username="admin", token="s3cret")
        self.control = handler.control_plane
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_reads_status_profiles_and_remotes_with_token(self) -> None:
        api = tray.ApiClient(self.url, token="s3cret")
        state = tray.fetch_state(api)
        self.assertIsNone(state.error)
        self.assertEqual(state.active["id"], "qwen-1")
        self.assertEqual(state.profiles[0]["mode"], "qwen")
        self.assertEqual(state.remotes[0]["name"], "oai")

    def test_surfaces_the_api_error_message(self) -> None:
        with self.assertRaisesRegex(tray.ApiError, "already has a tracked running server"):
            tray.ApiClient(self.url, token="s3cret").start("qwen")

    def test_wrong_token_and_unreachable_become_readable_errors(self) -> None:
        self.assertEqual(tray.fetch_state(tray.ApiClient(self.url, token="nope")).error, "unauthorized")
        state = tray.fetch_state(tray.ApiClient("http://127.0.0.1:1"))
        self.assertIn("not reachable", state.error)
        self.assertEqual(state.kind, "error")


class TrayStateTests(unittest.TestCase):
    def test_states(self) -> None:
        running = tray.TrayState(servers=[RUNNING])
        self.assertTrue(running.live)
        self.assertEqual(running.status_text(), "Running — qwen.gguf on :8080")
        paused = tray.TrayState(servers=[PAUSED])
        self.assertTrue(paused.paused and not paused.live)
        parked = tray.TrayState(servers=[PARKED])
        self.assertTrue(parked.parked and not parked.alive)
        self.assertEqual(parked.status_text(), "Released (GPU free) — qwen")
        self.assertEqual(tray.TrayState().status_text(), "Stopped — no tracked server")

    def test_running_server_wins_over_parked_record(self) -> None:
        state = tray.TrayState(servers=[PARKED, {**RUNNING, "id": "other"}])
        self.assertEqual(state.active["id"], "other")


class ControllerTests(unittest.TestCase):
    def _controller(self, servers):
        api = mock.Mock(spec=tray.ApiClient)
        api.action.return_value = {"success": True}
        api.start.return_value = {"success": True}
        api.remote.return_value = {"success": True}
        notes: list[str] = []
        c = tray.TrayController(api, notes.append, background=lambda fn: fn())
        c.state = tray.TrayState(servers=servers)
        c.refresh = mock.Mock()  # type: ignore[method-assign]
        return c, api, notes

    def test_context_preset_restarts_running_and_restores_parked(self) -> None:
        c, api, _ = self._controller([RUNNING])
        c.set_context(65536)
        api.action.assert_called_once_with("restart", "qwen-1", 65536)
        c, api, _ = self._controller([PARKED])
        c.set_context(16384)
        api.action.assert_called_once_with("restore", "qwen-1", 16384)

    def test_same_context_or_paused_does_nothing(self) -> None:
        for servers in ([RUNNING], [PAUSED]):
            c, api, _ = self._controller(servers)
            c.set_context(32768 if servers == [RUNNING] else 65536)
            api.action.assert_not_called()

    def test_failures_are_notified_and_state_refreshed(self) -> None:
        c, api, notes = self._controller([RUNNING])
        api.action.side_effect = tray.ApiError("needs AVX2, which this CPU lacks")
        c.act("restart")
        self.assertEqual(notes, ["needs AVX2, which this CPU lacks"])
        c.refresh.assert_called_once()

    def test_start_and_remote_toggle(self) -> None:
        c, api, notes = self._controller([])
        c.start("qwen")
        api.start.assert_called_once_with("qwen")
        self.assertEqual(notes, ["Starting qwen…"])
        c.toggle_remote({"name": "oai", "enabled": False})
        c.toggle_remote({"name": "oai", "enabled": True})
        self.assertEqual(api.remote.call_args_list, [mock.call("enable", "oai"), mock.call("disable")])


class StartWebTests(unittest.TestCase):
    def test_never_starts_a_remote_instance(self) -> None:
        with mock.patch.object(tray.subprocess, "Popen") as popen:
            self.assertFalse(tray.start_web_process("http://192.168.1.20:8716"))
        popen.assert_not_called()

    def test_starts_local_web_hidden_and_waits_for_it(self) -> None:
        with mock.patch.object(tray.subprocess, "Popen") as popen, \
             mock.patch.object(tray.ApiClient, "healthy", return_value=True), \
             mock.patch.object(tray, "NO_WINDOW", {"creationflags": 0x08000000}), \
             mock.patch.dict(os.environ, {"LCC_CACHE_DIR": self._tmp()}):
            self.assertTrue(tray.start_web_process("http://127.0.0.1:9123"))
        args, kwargs = popen.call_args
        self.assertEqual(args[0][1:], ["-m", "inferencedeck.webui", "--host", "127.0.0.1", "--port", "9123"])
        self.assertEqual(kwargs["creationflags"], 0x08000000)

    def _tmp(self) -> str:
        import tempfile

        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return d.name


class MenuTextTests(unittest.TestCase):
    def test_ampersands_are_doubled_only_on_windows(self) -> None:
        with mock.patch.object(tray, "is_windows", return_value=True):
            self.assertEqual(tray.menu_text("Reload & restart"), "Reload && restart")
        with mock.patch.object(tray, "is_windows", return_value=False):
            self.assertEqual(tray.menu_text("Reload & restart"), "Reload & restart")


@unittest.skipUnless(HAS_PYSTRAY, "pip install inferencedeck[tray]")
class PystrayMenuTests(unittest.TestCase):
    """Builds the real pystray menu on its display-less dummy backend."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["PYSTRAY_BACKEND"] = "dummy"
        import pystray

        cls.pystray = pystray

    def _menu(self, servers, profiles=(), remotes=()):
        api = mock.Mock(spec=tray.ApiClient)
        api.action.return_value = api.start.return_value = api.remote.return_value = {"success": True}
        c = tray.TrayController(api, lambda _m: None, background=lambda fn: fn())
        c.state = tray.TrayState(servers=list(servers), profiles=list(profiles), remotes=list(remotes))
        c.refresh = mock.Mock()  # type: ignore[method-assign]
        opened = []
        menu = tray.build_menu(self.pystray, c, lambda: opened.append(True), lambda: None)
        icon = self.pystray.Icon("t", tray.make_icon_image("idle"), "t", menu)
        items = {str(i.text): i for i in menu.items}
        return icon, items, api, opened

    def test_running_server_menu(self) -> None:
        icon, items, api, _ = self._menu([RUNNING])
        self.assertIn("Running — qwen.gguf on :8080", items)
        enabled = {name for name, item in items.items() if item.enabled}
        # Labels go through menu_text, which doubles "&" on Windows.
        expected = {"Pause (model stays in VRAM)", "Release GPU", tray.menu_text("Reload & restart"), "Context size", "Stop server"}
        self.assertTrue(expected <= enabled, enabled)
        self.assertFalse(items["Resume"].enabled or items["Restore"].enabled)
        sizes = {str(i.text): i for i in items["Context size"].submenu.items}
        self.assertTrue(sizes["32K"].checked and not sizes["64K"].checked)
        sizes["64K"](icon)
        api.action.assert_called_once_with("restart", "qwen-1", 65536)
        items["Release GPU"](icon)
        api.action.assert_called_with("release", "qwen-1", None)

    def test_parked_server_menu(self) -> None:
        icon, items, api, _ = self._menu([PARKED])
        self.assertIn("Forget released server", items)
        self.assertTrue(items["Restore"].enabled and not items["Pause (model stays in VRAM)"].enabled)
        items["Restore"](icon)
        api.action.assert_called_once_with("restore", "qwen-1", None)

    def test_profiles_remotes_and_web(self) -> None:
        profiles = [{"mode": "qwen", "name": "Qwen", "launchable": True, "model": {"name": "qwen.gguf"}},
                    {"mode": "bad", "name": "Bad", "launchable": False}]
        remotes = [{"name": "oai", "display_name": "OpenAI", "enabled": False, "selectable": False, "api_key_env": "OPENAI_API_KEY"}]
        icon, items, api, opened = self._menu([], profiles, remotes)
        start = {str(i.text): i for i in items["Start profile"].submenu.items}
        self.assertFalse(start["Bad — unresolved"].enabled)
        start["Qwen — qwen.gguf"](icon)
        api.start.assert_called_once_with("qwen")
        remote = [i for i in items[tray.menu_text("Remote & cloud models")].submenu.items if "OpenAI" in str(i.text)][0]
        self.assertEqual(str(remote.text), "OpenAI — set $OPENAI_API_KEY")
        self.assertFalse(remote.enabled)
        self.assertEqual(tray.remote_label({"display_name": "Qwen", "summary": "Thanatos · Tailscale · Self-hosted"}),
                         "Qwen (Thanatos · Tailscale · Self-hosted)")
        items["Open Web UI"](icon)
        self.assertEqual(opened, [True])
        self.assertTrue(items["Open Web UI"].default)

    def test_icon_image(self) -> None:
        for kind in ("running", "paused", "idle", "error"):
            self.assertEqual(tray.make_icon_image(kind).size, (64, 64))


if __name__ == "__main__":
    unittest.main()
