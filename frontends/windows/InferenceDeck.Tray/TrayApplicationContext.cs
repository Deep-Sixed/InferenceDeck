using System.Diagnostics;

namespace InferenceDeck.Tray;

internal sealed class TrayApplicationContext : ApplicationContext
{
    private readonly ControlClient _client;
    private readonly NotifyIcon _icon;
    private readonly ToolStripMenuItem _status = new("Connecting…") { Enabled = false };
    private readonly ToolStripMenuItem _profiles = new("Start profile");
    private readonly ToolStripMenuItem _suspend = new("Suspend (free GPU)");
    private readonly ToolStripMenuItem _resume = new("Resume server");
    private readonly ToolStripMenuItem _stop = new("Stop server");
    private readonly ToolStripMenuItem _command = new("Show active command");
    private readonly System.Windows.Forms.Timer _timer = new() { Interval = 5000 };
    private ServerState? _active;

    public TrayApplicationContext()
    {
        var controlUrl = Environment.GetEnvironmentVariable("INFERENCEDECK_CONTROL_URL") ?? "http://127.0.0.1:8717";
        _client = new ControlClient(controlUrl);
        var menu = new ContextMenuStrip();
        menu.Items.Add(_status);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add(_profiles);
        menu.Items.Add(_suspend);
        menu.Items.Add(_resume);
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
        _command.Click += (_, _) => MessageBox.Show(_active?.CommandLine ?? "No active command.", "InferenceDeck — Active command");
        _profiles.DropDownOpening += async (_, _) => await PopulateProfilesAsync();
        _timer.Tick += async (_, _) => await RefreshAsync();
        _timer.Start();
        _ = RefreshAsync();
    }

    private async Task RefreshAsync()
    {
        try
        {
            var servers = await _client.GetServersAsync();
            _active = servers.FirstOrDefault(s => s.Running);
            if (_active is null)
            {
                _status.Text = "Stopped — no tracked server";
                _icon.Text = "InferenceDeck — stopped";
            }
            else
            {
                var state = _active.Suspended ? "Suspended" : "Running";
                var model = Path.GetFileName(_active.ModelPath ?? _active.Mode);
                _status.Text = $"{state} — {model} on :{_active.Port}";
                _icon.Text = ($"InferenceDeck — {state}: {model}")[..Math.Min(63, $"InferenceDeck — {state}: {model}".Length)];
            }
            _suspend.Enabled = _active is { Running: true, Suspended: false };
            _resume.Enabled = _active is { Running: true, Suspended: true };
            _stop.Enabled = _active is { Running: true };
            _command.Enabled = _active is not null;
        }
        catch (Exception ex)
        {
            _status.Text = "Control API unavailable";
            _icon.Text = "InferenceDeck — API unavailable";
            _suspend.Enabled = _resume.Enabled = _stop.Enabled = false;
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
        var url = Environment.GetEnvironmentVariable("INFERENCEDECK_WEB_URL") ?? "http://127.0.0.1:8716";
        Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
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
