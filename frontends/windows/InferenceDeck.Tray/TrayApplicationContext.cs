using System.Diagnostics;

namespace InferenceDeck.Tray;

internal sealed class TrayApplicationContext : ApplicationContext
{
    private readonly ControlClient _client;
    private readonly NotifyIcon _icon;
    private readonly ToolStripMenuItem _status = new("Connecting…") { Enabled = false };
    private readonly ToolStripMenuItem _profiles = new("Start profile");
    private readonly ToolStripMenuItem _remotes = new("Remote & cloud models");
    private readonly ToolStripMenuItem _suspend = new("Pause (model stays in VRAM)");
    private readonly ToolStripMenuItem _resume = new("Resume");
    private readonly ToolStripMenuItem _release = new("Release GPU");
    private readonly ToolStripMenuItem _restore = new("Restore");
    private readonly ToolStripMenuItem _restart = new("Reload && restart");
    private readonly ToolStripMenuItem _context = new("Context size");
    private readonly ToolStripMenuItem _stop = new("Stop server");
    private readonly ToolStripMenuItem _command = new("Show active command");
    private readonly System.Windows.Forms.Timer _timer = new() { Interval = 5000 };
    private ServerState? _active;
    // inferencedeck-web serves both the API and the UI; it is the one process that owns server state.
    private static readonly string BaseUrl = Environment.GetEnvironmentVariable("INFERENCEDECK_URL") ?? "http://127.0.0.1:8716";

    public TrayApplicationContext()
    {
        _client = new ControlClient(BaseUrl);
        var menu = new ContextMenuStrip();
        menu.Items.Add(_status);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(_profiles);
        menu.Items.Add(_remotes);
        menu.Items.Add(_suspend);
        menu.Items.Add(_resume);
        menu.Items.Add(_release);
        menu.Items.Add(_restore);
        menu.Items.Add(_restart);
        menu.Items.Add(_context);
        menu.Items.Add(_stop);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Open Web UI", null, (_, _) => OpenWebUi());
        menu.Items.Add(_command);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Exit tray", null, (_, _) => ExitThread());

        _icon = new NotifyIcon
        {
            Text = "InferenceDeck",
            Icon = SystemIcons.Application,
            ContextMenuStrip = menu,
            Visible = true,
        };
        _icon.DoubleClick += async (_, _) => await ToggleSuspendAsync();
        _suspend.Click += async (_, _) => await ActAsync("Suspend", id => _client.SuspendAsync(id));
        _resume.Click += async (_, _) => await ActAsync("Resume", id => _client.ResumeAsync(id));
        _stop.Click += async (_, _) => await ActAsync("Stop", id => _client.StopAsync(id));
        _release.Click += async (_, _) => await ActAsync("Release GPU", id => _client.ReleaseAsync(id));
        _restore.Click += async (_, _) => await ActAsync("Restore", id => _client.RestoreAsync(id));
        _restart.Click += async (_, _) => await ActAsync("Reload & restart", id => _client.RestartAsync(id));
        foreach (var size in ControlClient.ContextPresets)
        {
            var item = new ToolStripMenuItem($"{size / 1024}K") { Tag = size };
            item.Click += async (_, _) =>
            {
                // A released server restores at the new size; a running one restarts at it.
                if (_active is { Parked: true }) await ActAsync("Restore", id => _client.RestoreAsync(id, size));
                else await ActAsync("Restart", id => _client.RestartAsync(id, size));
            };
            _context.DropDownItems.Add(item);
        }
        _command.Click += (_, _) => MessageBox.Show(_active?.CommandLine ?? "No active command.", "InferenceDeck — Active command");
        _profiles.DropDownOpening += async (_, _) => await PopulateProfilesAsync();
        _remotes.DropDownOpening += async (_, _) => await PopulateRemotesAsync();
        _timer.Tick += async (_, _) => await RefreshAsync();
        _timer.Start();
        _ = RefreshAsync();
    }

    private async Task RefreshAsync()
    {
        try
        {
            var servers = await _client.GetServersAsync();
            _active = servers.FirstOrDefault(s => s.Running) ?? servers.FirstOrDefault(s => s.Parked);
            if (_active is null)
            {
                _status.Text = "Stopped — no tracked server";
                _icon.Text = "InferenceDeck — stopped";
            }
            else
            {
                var state = _active.Parked ? "Released (GPU free)" : _active.Suspended ? "Paused" : "Running";
                var model = Path.GetFileName(_active.ModelPath ?? _active.Mode);
                _status.Text = $"{state} — {model} on :{_active.Port}";
                _icon.Text = ($"InferenceDeck — {state}: {model}")[..Math.Min(63, $"InferenceDeck — {state}: {model}".Length)];
            }
            var live = _active is { Running: true, Suspended: false };
            var parked = _active is { Parked: true };
            _suspend.Enabled = live;
            _resume.Enabled = _active is { Running: true, Suspended: true };
            _release.Enabled = _active is { Running: true };
            _restore.Enabled = parked;
            _restart.Enabled = live;
            _context.Enabled = live || parked;
            foreach (ToolStripMenuItem item in _context.DropDownItems)
                item.Checked = _active?.CtxSize is int ctx && item.Tag is int size && ctx == size;
            _stop.Enabled = _active is not null;
            _stop.Text = parked ? "Forget released server" : "Stop server";
            _command.Enabled = _active is { Running: true };
        }
        catch (Exception ex)
        {
            _status.Text = "Control API unavailable";
            _icon.Text = "InferenceDeck — API unavailable";
            _suspend.Enabled = _resume.Enabled = _release.Enabled = _restore.Enabled = false;
            _restart.Enabled = _context.Enabled = _stop.Enabled = false;
            Debug.WriteLine(ex);
        }
    }

    private async Task PopulateProfilesAsync()
    {
        _profiles.DropDownItems.Clear();
        try
        {
            foreach (var profile in await _client.GetProfilesAsync())
            {
                var item = new ToolStripMenuItem($"{profile.Name} — {profile.ModelName ?? "unresolved"}") { Enabled = profile.Launchable };
                item.Click += async (_, _) =>
                {
                    try { using var _ = await _client.StartAsync(profile.Mode); await RefreshAsync(); }
                    catch (Exception ex) { ShowError(ex); }
                };
                _profiles.DropDownItems.Add(item);
            }
            if (_profiles.DropDownItems.Count == 0) _profiles.DropDownItems.Add(new ToolStripMenuItem("No profiles found") { Enabled = false });
        }
        catch (Exception ex) { _profiles.DropDownItems.Add(new ToolStripMenuItem(ex.Message) { Enabled = false }); }
    }


    private async Task PopulateRemotesAsync()
    {
        _remotes.DropDownItems.Clear();
        try
        {
            var remotes = await _client.GetRemotesAsync();
            foreach (var remote in remotes)
            {
                var suffix = remote.Enabled ? " — active" : remote.Selectable ? "" : $" — set ${remote.ApiKeyEnv}";
                var item = new ToolStripMenuItem(remote.DisplayName + suffix) { Checked = remote.Enabled, Enabled = remote.Enabled || remote.Selectable };
                item.Click += async (_, _) =>
                {
                    try
                    {
                        using var _ = remote.Enabled ? await _client.DisableRemotesAsync() : await _client.EnableRemoteAsync(remote.Name);
                        await RefreshAsync();
                    }
                    catch (Exception ex) { ShowError(ex); }
                };
                _remotes.DropDownItems.Add(item);
            }
            if (remotes.Count > 0) _remotes.DropDownItems.Add(new ToolStripSeparator());
            var disable = new ToolStripMenuItem("Disable all remote/cloud models");
            disable.Click += async (_, _) => { try { using var _ = await _client.DisableRemotesAsync(); await RefreshAsync(); } catch (Exception ex) { ShowError(ex); } };
            _remotes.DropDownItems.Add(disable);
        }
        catch (Exception ex) { _remotes.DropDownItems.Add(new ToolStripMenuItem(ex.Message) { Enabled = false }); }
    }

    private async Task ToggleSuspendAsync()
    {
        if (_active is not { Running: true }) return;
        if (_active.Suspended) await ActAsync("Resume", id => _client.ResumeAsync(id));
        else await ActAsync("Suspend", id => _client.SuspendAsync(id));
    }

    private async Task ActAsync(string name, Func<string, Task<System.Text.Json.JsonDocument>> action)
    {
        if (_active is null) return;
        try { using var _ = await action(_active.Id); await RefreshAsync(); }
        catch (Exception ex) { ShowError(new InvalidOperationException($"{name} failed: {ex.Message}", ex)); }
    }

    private static void OpenWebUi()
    {
        Process.Start(new ProcessStartInfo(BaseUrl) { UseShellExecute = true });
    }

    private static void ShowError(Exception ex) => MessageBox.Show(ex.Message, "InferenceDeck", MessageBoxButtons.OK, MessageBoxIcon.Error);

    protected override void ExitThreadCore()
    {
        _timer.Stop();
        _icon.Visible = false;
        _icon.Dispose();
        base.ExitThreadCore();
    }
}
