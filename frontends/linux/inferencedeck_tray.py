#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from typing import Any

import gi

gi.require_version("AyatanaAppIndicator3", "0.1")
gi.require_version("Gtk", "3.0")
gi.require_version("Notify", "0.7")
from gi.repository import AyatanaAppIndicator3 as AppIndicator  # noqa: E402
from gi.repository import GLib, Gtk, Notify  # noqa: E402

# inferencedeck-web serves both the API and the UI; it is the one process that owns server state.
BASE_URL = os.environ.get("INFERENCEDECK_URL", "http://127.0.0.1:8716").rstrip("/")
TOKEN = os.environ.get("INFERENCEDECK_TOKEN", "").strip()
SYNC_SECONDS = 5
# start waits for the model to load before replying: up to 600 s server-side
# for MLC LLM (server_manager.READY_TIMEOUT_SECONDS), plus a minute of headroom
# for finding and launching the runtime.
START_TIMEOUT_SECONDS = 660
# stop allows 5 s for a clean exit plus 3 s after SIGKILL server-side.
STOP_TIMEOUT_SECONDS = 20
# restart stops and then starts, so it can take both.
RESTART_TIMEOUT_SECONDS = STOP_TIMEOUT_SECONDS + START_TIMEOUT_SECONDS
CONTEXT_PRESETS = (8192, 16384, 32768, 65536, 131072)


class ApiClient:
    def _request(self, path: str, body: dict[str, Any] | None = None, timeout: float = 5) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if TOKEN:
            headers["X-Auth-Token"] = TOKEN
        request = urllib.request.Request(
            BASE_URL + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Surface the API's own error message instead of "HTTP Error 400".
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except (ValueError, OSError):
                raise exc from None
            raise RuntimeError(payload.get("error") or payload.get("message") or str(exc)) from None

    def status(self) -> dict[str, Any]:
        return self._request("/api/status")

    def profiles(self) -> list[dict[str, Any]]:
        return self._request("/api/profiles").get("profiles", [])

    def remotes(self) -> dict[str, Any]:
        return self._request("/api/remotes")

    def start(self, mode: str) -> dict[str, Any]:
        return self._request("/api/start", {"mode": mode}, timeout=START_TIMEOUT_SECONDS)

    def action(self, action: str, server_id: str, ctx_size: int | None = None) -> dict[str, Any]:
        timeout = {
            "stop": STOP_TIMEOUT_SECONDS,
            "release": STOP_TIMEOUT_SECONDS,
            "restore": START_TIMEOUT_SECONDS,
            "restart": RESTART_TIMEOUT_SECONDS,
        }.get(action, 5)
        body: dict[str, Any] = {"server_id": server_id}
        if ctx_size is not None:
            body["ctx_size"] = ctx_size
        return self._request(f"/api/{action}", body, timeout=timeout)

    def remote(self, action: str, name: str = "") -> dict[str, Any]:
        return self._request("/api/remote", {"action": action, "name": name})


def _is_parked(server: dict[str, Any]) -> bool:
    # Released to free VRAM: no process, but Restore can start it again.
    return server.get("status") in ("parked", "restoring")


class TrayApplication:
    def __init__(self) -> None:
        self.api = ApiClient()
        self.active: dict[str, Any] | None = None
        self.remote_active = False
        self.status_item = Gtk.MenuItem(label="Connecting…")
        self.status_item.set_sensitive(False)
        self.profiles_item = Gtk.MenuItem(label="Start profile")
        self.profiles_menu = Gtk.Menu()
        self.profiles_item.set_submenu(self.profiles_menu)
        self.remotes_item = Gtk.MenuItem(label="Remote & cloud models")
        self.remotes_menu = Gtk.Menu()
        self.remotes_item.set_submenu(self.remotes_menu)
        self.suspend_item = Gtk.MenuItem(label="Pause (model stays in VRAM)")
        self.resume_item = Gtk.MenuItem(label="Resume")
        self.release_item = Gtk.MenuItem(label="Release GPU")
        self.restore_item = Gtk.MenuItem(label="Restore")
        self.restart_item = Gtk.MenuItem(label="Reload & restart")
        self.context_item = Gtk.MenuItem(label="Context size")
        context_menu = Gtk.Menu()
        self.context_items: dict[int, Gtk.CheckMenuItem] = {}
        for size in CONTEXT_PRESETS:
            item = Gtk.CheckMenuItem(label=f"{size // 1024}K")
            item.connect("activate", self._set_context, size)
            context_menu.append(item)
            self.context_items[size] = item
        self.context_item.set_submenu(context_menu)
        self.stop_item = Gtk.MenuItem(label="Stop server")
        self.command_item = Gtk.MenuItem(label="Show active command")
        self._syncing = False

        menu = Gtk.Menu()
        for item in [self.status_item, Gtk.SeparatorMenuItem(), self.profiles_item, self.remotes_item,
                     self.suspend_item, self.resume_item, self.release_item, self.restore_item,
                     self.restart_item, self.context_item, self.stop_item, Gtk.SeparatorMenuItem()]:
            menu.append(item)
        web = Gtk.MenuItem(label="Open Web UI")
        web.connect("activate", lambda *_: self._open_web())
        menu.append(web)
        menu.append(self.command_item)
        menu.append(Gtk.SeparatorMenuItem())
        quit_item = Gtk.MenuItem(label="Exit tray")
        quit_item.connect("activate", lambda *_: Gtk.main_quit())
        menu.append(quit_item)
        menu.show_all()

        self.suspend_item.connect("activate", lambda *_: self._act("suspend"))
        self.resume_item.connect("activate", lambda *_: self._act("resume"))
        self.stop_item.connect("activate", lambda *_: self._act("stop"))
        self.release_item.connect("activate", lambda *_: self._act("release"))
        self.restore_item.connect("activate", lambda *_: self._act("restore"))
        self.restart_item.connect("activate", lambda *_: self._act("restart"))
        self.command_item.connect("activate", lambda *_: self._show_command())
        self.profiles_menu.connect("show", self._populate_profiles)
        self.remotes_menu.connect("show", self._populate_remotes)

        self.indicator = AppIndicator.Indicator.new(
            "inferencedeck", "cpu", AppIndicator.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.indicator.set_menu(menu)
        self.indicator.set_title("InferenceDeck")
        Notify.init("InferenceDeck")
        GLib.timeout_add_seconds(SYNC_SECONDS, self._tick)
        self.refresh()

    def _notify(self, message: str, error: bool = False) -> None:
        n = Notify.Notification.new("InferenceDeck", message, "cpu")
        if error:
            n.set_urgency(Notify.Urgency.CRITICAL)
        n.show()

    def refresh(self) -> None:
        try:
            status = self.api.status()
            servers = status.get("servers", [])
            running = [s for s in servers if s.get("running")]
            parked_servers = [s for s in servers if _is_parked(s)]
            self.active = (running or parked_servers or [None])[0]
            self.remote_active = bool(status.get("remote_active"))
            if self.active:
                if _is_parked(self.active):
                    state = "Released (GPU free)"
                else:
                    state = "Paused" if self.active.get("suspended") else "Running"
                model = os.path.basename(str(self.active.get("model_path") or self.active.get("mode") or "server"))
                text = f"{state} — {model} on :{self.active.get('port', '—')}"
            elif self.remote_active:
                remote = status["remote_active"]
                text = f"Remote active — {remote.get('display_name') or remote.get('name')}"
            else:
                text = "Stopped — no tracked server"
            self.status_item.set_label(text)
            self.indicator.set_title(f"InferenceDeck — {text}")
            parked = bool(self.active and _is_parked(self.active))
            alive = bool(self.active and self.active.get("running"))
            suspended = bool(alive and self.active.get("suspended"))
            live = alive and not suspended
            self.suspend_item.set_sensitive(live)
            self.resume_item.set_sensitive(suspended)
            self.release_item.set_sensitive(alive)
            self.restore_item.set_sensitive(parked)
            self.restart_item.set_sensitive(live)
            self.context_item.set_sensitive(live or parked)
            current = int((self.active or {}).get("ctx_size") or 0)
            self._syncing = True  # set_active fires "activate"; don't treat it as a click
            try:
                for size, item in self.context_items.items():
                    item.set_active(size == current)
            finally:
                self._syncing = False
            self.stop_item.set_label("Forget released server" if parked else "Stop server")
            self.stop_item.set_sensitive(bool(self.active))
            self.command_item.set_sensitive(alive)
            self.profiles_item.set_sensitive(not self.remote_active)
        except Exception as exc:
            self.status_item.set_label("Control API unavailable")
            self.indicator.set_title("InferenceDeck — API unavailable")
            for item in [self.suspend_item, self.resume_item, self.release_item, self.restore_item,
                         self.restart_item, self.context_item, self.stop_item, self.command_item]:
                item.set_sensitive(False)
            sys.stderr.write(f"inferencedeck tray refresh: {exc}\n")

    def _tick(self) -> bool:
        self.refresh()
        return True

    def _clear(self, menu: Gtk.Menu) -> None:
        for item in list(menu.get_children()):
            menu.remove(item)

    def _populate_profiles(self, *_args) -> None:
        self._clear(self.profiles_menu)
        try:
            profiles = self.api.profiles()
            if not profiles:
                item = Gtk.MenuItem(label="No profiles found"); item.set_sensitive(False); self.profiles_menu.append(item)
            for profile in profiles:
                model = (profile.get("model") or {}).get("name") or "unresolved"
                item = Gtk.MenuItem(label=f"{profile.get('name') or profile.get('mode')} — {model}")
                item.set_sensitive(bool(profile.get("launchable")) and not self.remote_active)
                item.connect("activate", self._start_profile, str(profile.get("mode") or ""))
                self.profiles_menu.append(item)
        except Exception as exc:
            item = Gtk.MenuItem(label=str(exc)); item.set_sensitive(False); self.profiles_menu.append(item)
        self.profiles_menu.show_all()

    def _populate_remotes(self, *_args) -> None:
        self._clear(self.remotes_menu)
        try:
            endpoints = self.api.remotes().get("endpoints", [])
            for remote in endpoints:
                suffix = " — active" if remote.get("enabled") else ("" if remote.get("selectable") else f" — set ${remote.get('api_key_env')}")
                name = remote.get('display_name') or remote.get('name')
                summary = remote.get('summary')
                label = f"{name} ({summary})" if summary else f"{name}"
                item = Gtk.CheckMenuItem(label=f"{label}{suffix}")
                item.set_active(bool(remote.get("enabled")))
                item.set_sensitive(bool(remote.get("enabled") or remote.get("selectable")))
                item.connect("activate", self._toggle_remote, remote)
                self.remotes_menu.append(item)
            self.remotes_menu.append(Gtk.SeparatorMenuItem())
            off = Gtk.MenuItem(label="Disable all remote/cloud models")
            off.connect("activate", lambda *_: self._remote_action("disable"))
            self.remotes_menu.append(off)
        except Exception as exc:
            item = Gtk.MenuItem(label=str(exc)); item.set_sensitive(False); self.remotes_menu.append(item)
        self.remotes_menu.show_all()

    def _start_profile(self, _widget, mode: str) -> None:
        self._notify(f"Starting {mode}…")
        self._in_background(self.api.start, mode)

    def _act(self, action: str, ctx_size: int | None = None) -> None:
        if not self.active:
            return
        self._in_background(self.api.action, action, str(self.active.get("id")), ctx_size)

    def _set_context(self, _item: Gtk.CheckMenuItem, size: int) -> None:
        if self._syncing or not self.active:
            return
        if not _is_parked(self.active) and int(self.active.get("ctx_size") or 0) == size:
            self.refresh()  # re-check the current size; clicking it unchecked it
            return
        # A released server restores at the new size; a running one restarts at it.
        self._act("restore" if _is_parked(self.active) else "restart", size)

    def _in_background(self, call, *args) -> None:
        # Lifecycle calls can block for many seconds (start waits for the model
        # to load, stop for a clean exit); keep the GTK loop responsive.
        def worker() -> None:
            error = ""
            try:
                result = call(*args)
                if not result.get("success", True):
                    error = result.get("error") or result.get("message") or "request failed"
            except Exception as exc:
                error = str(exc)
            GLib.idle_add(self._background_done, error)

        threading.Thread(target=worker, daemon=True).start()

    def _background_done(self, error: str) -> bool:
        if error:
            self._notify(error, True)
        self.refresh()
        return False

    def _toggle_remote(self, widget: Gtk.CheckMenuItem, remote: dict[str, Any]) -> None:
        if widget.get_active() and not remote.get("enabled"):
            self._remote_action("enable", str(remote.get("name") or ""))
        elif not widget.get_active() and remote.get("enabled"):
            self._remote_action("disable")

    def _remote_action(self, action: str, name: str = "") -> None:
        self._in_background(self.api.remote, action, name)

    def _open_web(self) -> None:
        try:
            subprocess.Popen(["xdg-open", BASE_URL], start_new_session=True)
        except Exception as exc:
            self._notify(str(exc), True)

    def _show_command(self) -> None:
        command = str((self.active or {}).get("command_line") or "No active command.")
        dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.OK, text="Active command")
        dialog.format_secondary_text(command)
        dialog.run(); dialog.destroy()

    def run(self) -> None:
        Gtk.main()


def main() -> int:
    try:
        TrayApplication().run()
    except Exception as exc:
        print(f"inferencedeck-tray: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
