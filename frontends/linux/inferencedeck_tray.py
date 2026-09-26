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

CONTROL_URL = os.environ.get("INFERENCEDECK_CONTROL_URL", "http://127.0.0.1:8717").rstrip("/")
WEB_URL = os.environ.get("INFERENCEDECK_WEB_URL", "http://127.0.0.1:8716").rstrip("/")
TOKEN = os.environ.get("INFERENCEDECK_TOKEN", "").strip()
SYNC_SECONDS = 5
# start waits for the model to load (up to 45 s server-side) before replying.
START_TIMEOUT_SECONDS = 120


class ApiClient:
    def _request(self, path: str, body: dict[str, Any] | None = None, timeout: float = 5) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if TOKEN:
            headers["X-Auth-Token"] = TOKEN
        request = urllib.request.Request(
            CONTROL_URL + path,
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

    def action(self, action: str, server_id: str) -> dict[str, Any]:
        return self._request(f"/api/{action}", {"server_id": server_id})

    def remote(self, action: str, name: str = "") -> dict[str, Any]:
        return self._request("/api/remote", {"action": action, "name": name})


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
        self.suspend_item = Gtk.MenuItem(label="Suspend (free GPU)")
        self.resume_item = Gtk.MenuItem(label="Resume server")
        self.stop_item = Gtk.MenuItem(label="Stop server")
        self.command_item = Gtk.MenuItem(label="Show active command")

        menu = Gtk.Menu()
        for item in [self.status_item, Gtk.SeparatorMenuItem(), self.profiles_item, self.remotes_item,
                     self.suspend_item, self.resume_item, self.stop_item, Gtk.SeparatorMenuItem()]:
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
            running = [s for s in status.get("servers", []) if s.get("running")]
            self.active = running[0] if running else None
            self.remote_active = bool(status.get("remote_active"))
            if self.active:
                state = "Suspended" if self.active.get("suspended") else "Running"
                model = os.path.basename(str(self.active.get("model_path") or self.active.get("mode") or "server"))
                text = f"{state} — {model} on :{self.active.get('port', '—')}"
            elif self.remote_active:
                remote = status["remote_active"]
                text = f"Remote active — {remote.get('display_name') or remote.get('name')}"
            else:
                text = "Stopped — no tracked server"
            self.status_item.set_label(text)
            self.indicator.set_title(f"InferenceDeck — {text}")
            suspended = bool(self.active and self.active.get("suspended"))
            self.suspend_item.set_sensitive(bool(self.active) and not suspended)
            self.resume_item.set_sensitive(bool(self.active) and suspended)
            self.stop_item.set_sensitive(bool(self.active))
            self.command_item.set_sensitive(bool(self.active))
            self.profiles_item.set_sensitive(not self.remote_active)
        except Exception as exc:
            self.status_item.set_label("Control API unavailable")
            self.indicator.set_title("InferenceDeck — API unavailable")
            for item in [self.suspend_item, self.resume_item, self.stop_item, self.command_item]:
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
                item = Gtk.CheckMenuItem(label=f"{remote.get('display_name') or remote.get('name')}{suffix}")
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
        # Starting blocks until the model is loaded; keep the GTK loop responsive.
        self._notify(f"Starting {mode}…")
        threading.Thread(target=self._start_worker, args=(mode,), daemon=True).start()

    def _start_worker(self, mode: str) -> None:
        error = ""
        try:
            result = self.api.start(mode)
            if not result.get("success", True):
                error = result.get("error", "start failed")
        except Exception as exc:
            error = str(exc)
        GLib.idle_add(self._start_done, error)

    def _start_done(self, error: str) -> bool:
        if error:
            self._notify(error, True)
        self.refresh()
        return False

    def _act(self, action: str) -> None:
        if not self.active:
            return
        try:
            self.api.action(action, str(self.active.get("id")))
        except Exception as exc:
            self._notify(str(exc), True)
        self.refresh()

    def _toggle_remote(self, widget: Gtk.CheckMenuItem, remote: dict[str, Any]) -> None:
        if widget.get_active() and not remote.get("enabled"):
            self._remote_action("enable", str(remote.get("name") or ""))
        elif not widget.get_active() and remote.get("enabled"):
            self._remote_action("disable")

    def _remote_action(self, action: str, name: str = "") -> None:
        try:
            self.api.remote(action, name)
        except Exception as exc:
            self._notify(str(exc), True)
        self.refresh()

    def _open_web(self) -> None:
        try:
            subprocess.Popen(["xdg-open", WEB_URL], start_new_session=True)
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
