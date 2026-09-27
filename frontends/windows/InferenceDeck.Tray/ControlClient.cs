using System.Net.Http.Json;
using System.Text.Json;

namespace InferenceDeck.Tray;

internal sealed record ServerState(
    string Id,
    string Mode,
    int? Pid,
    bool Running,
    bool Suspended,
    string? CommandLine,
    string? ModelPath,
    string? Host,
    int? Port);

internal sealed record ProfileState(string Mode, string Name, bool Launchable, string? ModelName);

internal sealed class ControlClient
{
    private readonly HttpClient _http;

    public ControlClient(string baseUrl)
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/"), Timeout = TimeSpan.FromSeconds(5) };
    }

    public async Task<IReadOnlyList<ServerState>> GetServersAsync()
    {
        using var doc = await GetJsonAsync("api/status");
        var values = new List<ServerState>();
        if (!doc.RootElement.TryGetProperty("servers", out var servers)) return values;
        foreach (var item in servers.EnumerateArray())
        {
            values.Add(new ServerState(
                Text(item, "id"), Text(item, "mode"), Int(item, "pid"), Bool(item, "running"), Bool(item, "suspended"),
                TextOrNull(item, "command_line"), TextOrNull(item, "model_path"), TextOrNull(item, "host"), Int(item, "port")));
        }
        return values;
    }

    public async Task<IReadOnlyList<ProfileState>> GetProfilesAsync()
    {
        using var doc = await GetJsonAsync("api/profiles");
        var values = new List<ProfileState>();
        if (!doc.RootElement.TryGetProperty("profiles", out var profiles)) return values;
        foreach (var item in profiles.EnumerateArray())
        {
            string? modelName = null;
            if (item.TryGetProperty("model", out var model) && model.ValueKind == JsonValueKind.Object)
                modelName = TextOrNull(model, "name");
            values.Add(new ProfileState(Text(item, "mode"), Text(item, "name"), Bool(item, "launchable"), modelName));
        }
        return values;
    }

    public Task<JsonDocument> StartAsync(string mode) => PostAsync("api/start", new { mode });
    public Task<JsonDocument> StopAsync(string serverId) => PostAsync("api/stop", new { server_id = serverId });
    public Task<JsonDocument> SuspendAsync(string serverId) => PostAsync("api/suspend", new { server_id = serverId });
    public Task<JsonDocument> ResumeAsync(string serverId) => PostAsync("api/resume", new { server_id = serverId });

    private async Task<JsonDocument> GetJsonAsync(string path)
    {
        using var response = await _http.GetAsync(path);
        var bytes = await response.Content.ReadAsByteArrayAsync();
        response.EnsureSuccessStatusCode();
        return JsonDocument.Parse(bytes);
    }

    private async Task<JsonDocument> PostAsync(string path, object body)
    {
        using var response = await _http.PostAsJsonAsync(path, body);
        var bytes = await response.Content.ReadAsByteArrayAsync();
        response.EnsureSuccessStatusCode();
        return JsonDocument.Parse(bytes);
    }

    private static string Text(JsonElement item, string name) => TextOrNull(item, name) ?? string.Empty;
    private static string? TextOrNull(JsonElement item, string name) => item.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
    private static bool Bool(JsonElement item, string name) => item.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False && v.GetBoolean();
    private static int? Int(JsonElement item, string name) => item.TryGetProperty(name, out var v) && v.TryGetInt32(out var value) ? value : null;
}
