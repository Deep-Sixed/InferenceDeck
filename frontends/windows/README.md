# InferenceDeck Windows tray

Native Windows system-tray frontend for the InferenceDeck control plane, derived from the proven Thanatos `LlamaServerTray` interaction model.

The tray intentionally contains **no model-discovery or server-launch logic**. It talks to the local InferenceDeck control API (default `http://127.0.0.1:8717`) and opens the web UI at `http://127.0.0.1:8716`.

Environment overrides:

- `INFERENCEDECK_CONTROL_URL`
- `INFERENCEDECK_WEB_URL`

Build on Windows with .NET 8:

```powershell
dotnet build .\frontends\windows\InferenceDeck.Tray\InferenceDeck.Tray.csproj -c Release
```
