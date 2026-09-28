# InferenceDeck Linux tray

GTK/AyatanaAppIndicator frontend derived from the proven EVECOR Linux tray interaction model. This frontend is deliberately thin: it does not manage systemd, processes, models, or cloud configuration directly. All state and actions go through the InferenceDeck control API.

## Dependencies

On Debian/Ubuntu:

```bash
sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1 gir1.2-gtk-3.0 gir1.2-notify-0.7
```

The distro/system Python is normally required because PyGObject is provided by the OS packages.

## Environment

- `INFERENCEDECK_URL` — the `inferencedeck-web` process, default `http://127.0.0.1:8716` (serves both the API and the web UI)
- `INFERENCEDECK_TOKEN` or `INFERENCEDECK_TOKEN_FILE` — required when authentication is enabled; read the same way as the server
  reads them. After a rejected token the tray checks again less often (up to every 10 minutes), so a stale
  token can't lock this machine out of the web UI.

## Runtime updates

The **Runtime updates** menu entry says how many runtimes have a newer release, checked at
startup and then hourly. Its submenu lists each one (clicking opens the GitHub release page)
and has **Check now**. InferenceDeck never downloads or installs an update itself.

## Run

```bash
/usr/bin/python3 frontends/linux/inferencedeck_tray.py
```

The supplied `.desktop` file is a template; replace `/path/to/InferenceDeck` with the checkout/install path before placing it in `~/.config/autostart/`.
