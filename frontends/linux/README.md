# InferenceDeck Linux tray

GTK/AyatanaAppIndicator frontend derived from the proven EVECOR Linux tray interaction model. This frontend is deliberately thin: it does not manage systemd, processes, models, or cloud configuration directly. All state and actions go through the InferenceDeck control API.

## Dependencies

On Debian/Ubuntu:

```bash
sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1 gir1.2-gtk-3.0 gir1.2-notify-0.7
```

The distro/system Python is normally required because PyGObject is provided by the OS packages.

## Environment

- `INFERENCEDECK_CONTROL_URL` — default `http://127.0.0.1:8717`
- `INFERENCEDECK_WEB_URL` — default `http://127.0.0.1:8716`
- `INFERENCEDECK_TOKEN` — required when the control API has authentication enabled

## Run

```bash
/usr/bin/python3 frontends/linux/inferencedeck_tray.py
```

The supplied `.desktop` file is a template; replace `/path/to/InferenceDeck` with the checkout/install path before placing it in `~/.config/autostart/`.
