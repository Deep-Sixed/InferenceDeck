# InferenceDeck tray (Windows and macOS)

A system-tray frontend written in Python with [pystray](https://github.com/moses-palmer/pystray).
It replaces the earlier .NET tray, so there is no .NET runtime to install.

Like every InferenceDeck frontend it holds **no model-discovery or server-launch logic**:
it talks to the `inferencedeck-web` process (default `http://127.0.0.1:8716`), which
serves the control API and the web UI.

## Install and run

```powershell
pip install "inferencedeck[tray]"
inferencedeck-tray
```

On Windows `inferencedeck-tray.exe` is a GUI program (no console window). If
`inferencedeck-web` isn't running on this machine, the tray starts it in the background
(hidden) and waits for it. Its output goes to the InferenceDeck cache folder's `logs\web.log`.

## Menu

Status · Open Web UI (also the tray icon's default click) · Start profile · Remote & cloud
models · Pause / Resume · Release GPU / Restore · Reload & restart · Context size
(8K-128K) · Stop / Forget released server · Copy active command · Exit.

The icon colour follows the state: green running, amber paused, grey idle or released,
red when InferenceDeck can't be reached.

## Settings (environment variables)

- `INFERENCEDECK_URL`: the `inferencedeck-web` address, default `http://127.0.0.1:8716`.
- `INFERENCEDECK_TOKEN` or `INFERENCEDECK_TOKEN_FILE`: required when authentication is enabled; read the
  same way as the server reads them. After a rejected token the tray checks again less often (up to every
  10 minutes), so a stale token can't lock this machine out of the web UI.
- `INFERENCEDECK_TRAY_START_WEB=0`: don't start `inferencedeck-web` automatically. It is
  only ever started for a local (loopback) address.

## Start at logon (Windows)

Press Win+R, open `shell:startup`, and create a shortcut there to `inferencedeck-tray.exe`
(it is in your Python environment's `Scripts` folder: `where inferencedeck-tray`).

Linux uses the GTK/AppIndicator tray in `frontends/linux/`.
