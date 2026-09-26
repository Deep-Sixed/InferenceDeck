# InferenceDeck Windows tray

Native Windows system-tray frontend for the InferenceDeck control plane, derived from the proven Thanatos `LlamaServerTray` interaction model.

The tray intentionally contains **no model-discovery or server-launch logic**. It talks to the `inferencedeck-web` process (default `http://127.0.0.1:8716`), which serves both the control API and the web UI.

Environment overrides:

- `INFERENCEDECK_URL` — the `inferencedeck-web` address
- `INFERENCEDECK_TOKEN` — required when authentication is enabled

Build on Windows with .NET 8:

```powershell
dotnet build .\frontends\windows\InferenceDeck.Tray\InferenceDeck.Tray.csproj -c Release
```
