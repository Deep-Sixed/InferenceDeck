"""InferenceDeck system tray for Windows and macOS.

Replaces the .NET tray: ``pip install inferencedeck[tray]`` then run
``inferencedeck-tray`` (a GUI program on Windows, so no console window and no
.NET runtime). Linux keeps its GTK/AppIndicator tray in ``frontends/linux``.

Like every InferenceDeck frontend, the tray holds no inference logic: it talks
to the ``inferencedeck-web`` process, and starts it (hidden) if it isn't
running yet.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from .auth import LOOPBACK_HOSTS
from .paths import cache_dir, is_windows
from .proc import NO_WINDOW, run as run_hidden

BASE_URL = os.environ.get("INFERENCEDECK_URL", "http://127.0.0.1:8716").rstrip("/")
TOKEN = os.environ.get("INFERENCEDECK_TOKEN", "").strip()
POLL_SECONDS = 5
CONTEXT_PRESETS = (8192, 16384, 32768, 65536, 131072)
# Server-side waits: start waits for the model to load (up to 180 s for
# vllm.cpp, see server_manager.READY_TIMEOUT_SECONDS) after finding the runtime
# and launching it; stop allows 5 s for a clean exit plus 3 s after a forced
# kill; restart does both. Each client timeout is the server's longest wait
# plus a minute of headroom, so the tray never gives up on a start that is
# still going to succeed.
START_TIMEOUT_SECONDS = 240
STOP_TIMEOUT_SECONDS = 20
TIMEOUTS = {
    "start": START_TIMEOUT_SECONDS,
    "restore": START_TIMEOUT_SECONDS,
    "restart": STOP_TIMEOUT_SECONDS + START_TIMEOUT_SECONDS,
    "stop": STOP_TIMEOUT_SECONDS,
    "release": STOP_TIMEOUT_SECONDS,
}


class ApiError(Exception):
    """A failed API call, carrying the API's own error message."""


class ApiClient:
    def __init__(self, base_url: str = BASE_URL, token: str = TOKEN) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(self, path: str, body: dict[str, Any] | None = None, timeout: float = 5) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["X-Auth-Token"] = self.token
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                message = payload.get("error") or payload.get("message")
            except (ValueError, OSError, AttributeError):
                message = None
            raise ApiError(message or f"HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ApiError(f"InferenceDeck is not reachable at {self.base_url} ({exc})") from None

    def healthy(self) -> bool:
        try:
            return bool(self.request("/healthz", timeout=2).get("ok"))
        except ApiError:
            return False

    def status(self) -> dict[str, Any]:
        return self.request("/api/status")

    def profiles(self) -> list[dict[str, Any]]:
        return self.request("/api/profiles").get("profiles", [])

    def remotes(self) -> list[dict[str, Any]]:
        return self.request("/api/remotes").get("endpoints", [])

    def start(self, mode: str) -> dict[str, Any]:
        return self.request("/api/start", {"mode": mode}, timeout=TIMEOUTS["start"])

    def action(self, action: str, server_id: str, ctx_size: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"server_id": server_id}
        if ctx_size is not None:
            body["ctx_size"] = ctx_size
        return self.request(f"/api/{action}", body, timeout=TIMEOUTS.get(action, 5))

    def remote(self, action: str, name: str = "") -> dict[str, Any]:
        return self.request("/api/remote", {"action": action, "name": name})


def is_parked(server: dict[str, Any]) -> bool:
    # Released to free VRAM: no process, but Restore can start it again.
    return server.get("status") in ("parked", "restoring")


@dataclass
class TrayState:
    servers: list[dict[str, Any]] = field(default_factory=list)
    profiles: list[dict[str, Any]] = field(default_factory=list)
    remotes: list[dict[str, Any]] = field(default_factory=list)
    remote_active: dict[str, Any] | None = None
    error: str | None = None

    @property
    def active(self) -> dict[str, Any] | None:
        running = [s for s in self.servers if s.get("running")]
        parked = [s for s in self.servers if is_parked(s)]
        return (running or parked or [None])[0]

    @property
    def alive(self) -> bool:
        return bool(self.active and self.active.get("running"))

    @property
    def paused(self) -> bool:
        return self.alive and bool(self.active.get("suspended"))  # type: ignore[union-attr]

    @property
    def live(self) -> bool:
        return self.alive and not self.paused

    @property
    def parked(self) -> bool:
        return bool(self.active and is_parked(self.active))

    @property
    def ctx_size(self) -> int:
        return int((self.active or {}).get("ctx_size") or 0)

    @property
    def kind(self) -> str:
        """Icon colour: error, running, paused or idle."""
        if self.error:
            return "error"
        if self.live:
            return "running"
        if self.paused:
            return "paused"
        return "idle"

    def status_text(self) -> str:
        if self.error:
            return "InferenceDeck unavailable"
        server = self.active
        if server:
            state = "Released (GPU free)" if self.parked else "Paused" if self.paused else "Running"
            model = os.path.basename(str(server.get("model_path") or server.get("mode") or "server"))
            port = f" on :{server['port']}" if server.get("port") and not self.parked else ""
            return f"{state} — {model}{port}"
        if self.remote_active:
            return f"Remote active — {self.remote_active.get('display_name') or self.remote_active.get('name')}"
        return "Stopped — no tracked server"


def fetch_state(api: ApiClient) -> TrayState:
    try:
        status = api.status()
        return TrayState(
            servers=list(status.get("servers") or []),
            profiles=api.profiles(),
            remotes=api.remotes(),
            remote_active=status.get("remote_active"),
        )
    except ApiError as exc:
        return TrayState(error=str(exc))


class TrayController:
    """Menu actions and state, independent of the tray toolkit."""

    def __init__(
        self,
        api: ApiClient,
        notify: Callable[[str], None],
        on_change: Callable[[], None] = lambda: None,
        background: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.api = api
        self.notify = notify
        self.on_change = on_change
        self.background = background or (lambda fn: threading.Thread(target=fn, daemon=True).start())
        self.state = TrayState(error="connecting")

    def refresh(self) -> None:
        self.state = fetch_state(self.api)
        self.on_change()

    def _run(self, call: Callable[..., dict[str, Any]], *args: Any) -> None:
        # Lifecycle calls block for seconds (start waits for the model to load),
        # so they never run on the tray's UI thread.
        def work() -> None:
            try:
                result = call(*args)
                if not result.get("success", True):
                    self.notify(result.get("error") or result.get("message") or "Request failed")
            except ApiError as exc:
                self.notify(str(exc))
            self.refresh()

        self.background(work)

    def start(self, mode: str) -> None:
        self.notify(f"Starting {mode}…")
        self._run(self.api.start, mode)

    def act(self, action: str, ctx_size: int | None = None) -> None:
        server = self.state.active
        if server:
            self._run(self.api.action, action, str(server.get("id")), ctx_size)

    def set_context(self, size: int) -> None:
        if self.state.parked:
            self.act("restore", size)
        elif self.state.live and self.state.ctx_size != size:
            self.act("restart", size)

    def toggle_remote(self, remote: dict[str, Any]) -> None:
        if remote.get("enabled"):
            self._run(self.api.remote, "disable")
        else:
            self._run(self.api.remote, "enable", str(remote.get("name") or ""))

    def disable_remotes(self) -> None:
        self._run(self.api.remote, "disable")

    def active_command(self) -> str | None:
        return (self.state.active or {}).get("command_line") if self.state.alive else None


def copy_to_clipboard(text: str) -> bool:
    try:
        if is_windows():
            result = run_hidden(["clip"], input=text, text=True, timeout=5, check=False)
        elif sys.platform == "darwin":
            result = run_hidden(["pbcopy"], input=text, text=True, timeout=5, check=False)
        else:
            return False
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def start_web_process(base_url: str = BASE_URL) -> bool:
    """Start inferencedeck-web in the background when it isn't running locally.

    Only for a loopback URL: a remote InferenceDeck is someone else's to start.
    Returns True once it answers, False if not started or not up within 20 s.
    """

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if host.lower() not in LOOPBACK_HOSTS:
        return False
    python = sys.executable
    if is_windows():
        # pythonw has no console, so nothing flashes even for a moment.
        pythonw = os.path.join(os.path.dirname(python), "pythonw.exe")
        python = pythonw if os.path.isfile(pythonw) else python
    log_path = cache_dir() / "logs" / "web.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        subprocess.Popen(
            [python, "-m", "inferencedeck.webui", "--host", host, "--port", str(parsed.port or 8716)],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            creationflags=NO_WINDOW.get("creationflags", 0),  # CREATE_NO_WINDOW on Windows
        )
    api = ApiClient(base_url)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if api.healthy():
            return True
        time.sleep(0.5)
    return False


ICON_COLOURS = {"running": "#37b66b", "paused": "#d9a441", "idle": "#6b7a8f", "error": "#d95b65"}


def make_icon_image(kind: str):
    from PIL import Image, ImageDraw

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, size - 3, size - 3), radius=14, fill=ICON_COLOURS.get(kind, ICON_COLOURS["idle"]))
    # Three "deck" bars.
    for i, width in enumerate((40, 32, 24)):
        top = 16 + i * 12
        draw.rounded_rectangle((12, top, 12 + width, top + 7), radius=3, fill="white")
    return image


def menu_text(text: str) -> str:
    # Windows menus treat "&" as a keyboard-accelerator marker; "&&" shows one "&".
    return text.replace("&", "&&") if is_windows() else text


def build_menu(pystray: Any, controller: TrayController, open_web: Callable[[], None], quit_tray: Callable[[], None]):
    Item, Menu = pystray.MenuItem, pystray.Menu
    st = lambda: controller.state  # noqa: E731 - always read the latest state

    # pystray picks how to call an action from its argument count, defaults
    # included, so per-item values are bound by closures, not default args.
    def start_action(mode: str) -> Callable[[], None]:
        return lambda: controller.start(mode)

    def remote_action(remote: dict[str, Any]) -> Callable[[], None]:
        return lambda: controller.toggle_remote(remote)

    def context_action(size: int) -> Callable[[], None]:
        return lambda: controller.set_context(size)

    def context_checked(size: int) -> Callable[[Any], bool]:
        return lambda _item: st().ctx_size == size

    def profile_items():
        profiles = st().profiles
        if not profiles:
            yield Item("No profiles found", None, enabled=False)
        for p in profiles:
            label = f"{p.get('name') or p.get('mode')} — {(p.get('model') or {}).get('name') or 'unresolved'}"
            yield Item(
                menu_text(label),
                start_action(str(p.get("mode"))),
                enabled=bool(p.get("launchable")) and not st().remote_active,
            )

    def remote_items():
        for r in st().remotes:
            suffix = " — active" if r.get("enabled") else "" if r.get("selectable") else f" — set ${r.get('api_key_env')}"
            yield Item(
                menu_text(f"{r.get('display_name') or r.get('name')}{suffix}"),
                remote_action(r),
                checked=lambda _item, on=bool(r.get("enabled")): on,
                enabled=bool(r.get("enabled") or r.get("selectable")),
            )
        yield Menu.SEPARATOR
        yield Item("Disable all remote/cloud models", lambda: controller.disable_remotes())

    def context_items():
        for size in CONTEXT_PRESETS:
            yield Item(
                f"{size // 1024}K",
                context_action(size),
                checked=context_checked(size),
                radio=True,
            )

    def copy_command() -> None:
        command = controller.active_command()
        if command and copy_to_clipboard(command):
            controller.notify("Active command copied to the clipboard.")
        elif command:
            controller.notify(command)

    return Menu(
        Item(lambda _item: menu_text(st().status_text()), None, enabled=False),
        Menu.SEPARATOR,
        Item("Open Web UI", lambda: open_web(), default=True),
        Menu.SEPARATOR,
        Item("Start profile", Menu(profile_items)),
        Item(menu_text("Remote & cloud models"), Menu(remote_items)),
        Item("Pause (model stays in VRAM)", lambda: controller.act("suspend"), enabled=lambda _i: st().live),
        Item("Resume", lambda: controller.act("resume"), enabled=lambda _i: st().paused),
        Item("Release GPU", lambda: controller.act("release"), enabled=lambda _i: st().alive),
        Item("Restore", lambda: controller.act("restore"), enabled=lambda _i: st().parked),
        Item(menu_text("Reload & restart"), lambda: controller.act("restart"), enabled=lambda _i: st().live),
        Item("Context size", Menu(context_items), enabled=lambda _i: st().live or st().parked),
        Item(
            lambda _i: "Forget released server" if st().parked else "Stop server",
            lambda: controller.act("stop"),
            enabled=lambda _i: st().active is not None,
        ),
        Menu.SEPARATOR,
        Item("Copy active command", copy_command, enabled=lambda _i: st().alive),
        Menu.SEPARATOR,
        Item("Exit tray", lambda: quit_tray()),
    )


def main() -> int:
    try:
        import pystray
    except ImportError:
        sys.stderr.write("The tray needs extra packages: pip install inferencedeck[tray]\n")
        return 1

    api = ApiClient()
    if not api.healthy() and os.environ.get("INFERENCEDECK_TRAY_START_WEB", "1") != "0":
        start_web_process()

    icon = pystray.Icon("inferencedeck", make_icon_image("idle"), "InferenceDeck")
    stop = threading.Event()

    def notify(message: str) -> None:
        try:
            icon.notify(message, "InferenceDeck")
        except Exception:  # notifications are best effort
            pass

    def on_change() -> None:
        state = controller.state
        icon.icon = make_icon_image(state.kind)
        # Plain ASCII: some tray backends (X11) can't encode other characters in a tooltip.
        icon.title = f"InferenceDeck - {state.status_text()}".replace("\u2014", "-")[:127]
        icon.update_menu()

    controller = TrayController(api, notify, on_change)

    def quit_tray() -> None:
        stop.set()
        icon.stop()

    icon.menu = build_menu(pystray, controller, lambda: webbrowser.open(api.base_url), quit_tray)

    def poll(_icon: Any) -> None:
        _icon.visible = True
        while not stop.is_set():
            try:
                controller.refresh()
            except Exception as exc:  # one bad update must not freeze the tray for good
                sys.stderr.write(f"inferencedeck tray refresh failed: {exc}\n")
            stop.wait(POLL_SECONDS)

    icon.run(setup=poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
