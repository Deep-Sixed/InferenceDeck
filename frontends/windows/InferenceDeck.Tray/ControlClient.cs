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
    int? Port,
    string? Status,
    int? CtxSize)
{
    // Released to free VRAM: no process, but Restore can start it again.
    public bool Parked => Status is "parked" or "restoring";
}

internal sealed record ProfileState(string Mode, string Name, bool Launchable, string? ModelName);
internal sealed record RemoteState(string Name, string DisplayName, string Provider, string Lane, bool Enabled, bool Selectable, string ApiKeyEnv);

internal sealed class ControlClient
{
    private static readonly TimeSpan ShortTimeout = TimeSpan.FromSeconds(5);
    // api/start waits for the model to load (up to 45 s server-side) before replying.
    private static readonly TimeSpan StartTimeout = TimeSpan.FromSeconds(120);
    // api/stop allows 5 s for a clean exit plus 3 s after a forced kill.
    private static readonly TimeSpan StopTimeout = TimeSpan.FromSeconds(20);
    // api/restart stops and then starts, so it can take both.
    private static readonly TimeSpan RestartTimeout = StopTimeout + StartTimeout;
    public static readonly int[] ContextPresets = { 8192, 16384, 32768, 65536, 131072 };
    private readonly HttpClient _http;

    public ControlClient(string baseUrl)
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/"), Timeout = System.Threading.Timeout.InfiniteTimeSpan };
        var token = Environment.GetEnvironmentVariable("INFERENCEDECK_TOKEN");
        if (!string.IsNullOrWhiteSpace(token)) _http.DefaultRequestHeaders.Add("X-Auth-Token", token);
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
                TextOrNull(item, "command_line"), TextOrNull(item, "model_path"), TextOrNull(item, "host"), Int(item, "port"),
                TextOrNull(item, "status"), Int(item, "ctx_size")));
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

    public async Task<IReadOnlyList<RemoteState>> GetRemotesAsync()
    {
        using var doc = await GetJsonAsync("api/remotes");
        var values = new List<RemoteState>();
        if (!doc.RootElement.TryGetProperty("endpoints", out var endpoints)) return values;
        foreach (var item in endpoints.EnumerateArray())
            values.Add(new RemoteState(Text(item,"name"), Text(item,"display_name"), Text(item,"provider"), Text(item,"lane"), Bool(item,"enabled"), Bool(item,"selectable"), Text(item,"api_key_env")));
        return values;
    }

    public Task<JsonDocument> EnableRemoteAsync(string name) => PostAsync("api/remote", new { action = "enable", name });
    public Task<JsonDocument> DisableRemotesAsync() => PostAsync("api/remote", new { action = "disable" });

    public Task<JsonDocument> StartAsync(string mode) => PostAsync("api/start", new { mode }, StartTimeout);
    public Task<JsonDocument> StopAsync(string serverId) => PostAsync("api/stop", new { server_id = serverId }, StopTimeout);
    public Task<JsonDocument> SuspendAsync(string serverId) => PostAsync("api/suspend", new { server_id = serverId });
    public Task<JsonDocument> ResumeAsync(string serverId) => PostAsync("api/resume", new { server_id = serverId });
    public Task<JsonDocument> ReleaseAsync(string serverId) => PostAsync("api/release", new { server_id = serverId }, StopTimeout);
    public Task<JsonDocument> RestoreAsync(string serverId, int? ctxSize = null) =>
        PostAsync("api/restore", new { server_id = serverId, ctx_size = ctxSize }, StartTimeout);
    public Task<JsonDocument> RestartAsync(string serverId, int? ctxSize = null) =>
        PostAsync("api/restart", new { server_id = serverId, ctx_size = ctxSize }, RestartTimeout);

    private async Task<JsonDocument> GetJsonAsync(string path)
    {
        using var cts = new CancellationTokenSource(ShortTimeout);
        using var response = await _http.GetAsync(path, cts.Token);
        return await ReadAsync(response, cts.Token);
    }

    private async Task<JsonDocument> PostAsync(string path, object body, TimeSpan? timeout = null)
    {
        using var cts = new CancellationTokenSource(timeout ?? ShortTimeout);
        using var response = await _http.PostAsJsonAsync(path, body, cts.Token);
        return await ReadAsync(response, cts.Token);
    }

    private static async Task<JsonDocument> ReadAsync(HttpResponseMessage response, CancellationToken token)
    {
        var bytes = await response.Content.ReadAsByteArrayAsync(token);
        if (response.IsSuccessStatusCode) return JsonDocument.Parse(bytes);
        // Surface the API's own error message instead of "400 (Bad Request)".
        string? error = null;
        try
        {
            using var doc = JsonDocument.Parse(bytes);
            error = TextOrNull(doc.RootElement, "error") ?? TextOrNull(doc.RootElement, "message");
        }
        catch (JsonException) { }
        throw new HttpRequestException(error ?? $"{(int)response.StatusCode} {response.ReasonPhrase}", null, response.StatusCode);
    }

    private static string Text(JsonElement item, string name) => TextOrNull(item, name) ?? string.Empty;
    private static string? TextOrNull(JsonElement item, string name) => item.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
    private static bool Bool(JsonElement item, string name) => item.TryGetProperty(name, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False && v.GetBoolean();
    private static int? Int(JsonElement item, string name) => item.TryGetProperty(name, out var v) && v.TryGetInt32(out var value) ? value : null;
}
