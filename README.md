# InferenceDeck

Cross-platform control plane for local AI inference — discover runtimes and GGUF models, detect hardware, estimate fit, manage profiles, benchmark performance, and control local or remote inference endpoints.

## Current scope

InferenceDeck is a clean continuation of the portable core developed in the earlier local **Llama Control Center** project. Historical Windows and Linux control surfaces were used as donor implementations, but machine-specific paths, service files, compiled runtimes, and private endpoint configuration are intentionally excluded.

### Core

- Discover local `llama.cpp`, `vllm.cpp`, Ollama, LM Studio, vLLM, and MLX runtimes.
- Discover GGUF models from configured and common model locations.
- Detect CPU, GPU, system memory, VRAM, and available acceleration backends.
- Estimate model fit and performance, with `llama-fit-params` integration when available.
- Resolve and manage portable model profiles.
- Prepare `llama-server` or `vllm-server` (vllm.cpp) launch commands and manage servers started by InferenceDeck.
- Pause/resume tracked servers without losing process state, or release the GPU (stop the server, keep its settings) and restore it later.
- Benchmark local OpenAI-compatible inference endpoints and retain bounded benchmark history.
- Inspect Hugging Face tooling and runtime update availability.
- Generate portable launch scripts without overwriting hand-written scripts.

### Frontends

- Local browser control panel (`inferencedeck-web`).
- Python system tray for Windows and macOS (`inferencedeck-tray`, see `frontends/windows/`). No .NET runtime needed.
- Linux GTK/AyatanaAppIndicator tray under `frontends/linux/`.
- All frontends use the same InferenceDeck control API; they do not duplicate model discovery or server-management logic.

### Remote and cloud endpoints

InferenceDeck supports two endpoint lanes:

- `remote_host` — another self-hosted runtime, such as `llama.cpp` over a LAN or tailnet.
- `true_cloud` — a hosted API endpoint.

Endpoint definitions live under the per-user InferenceDeck configuration directory in `remote_endpoints/*.json`. Generic examples are provided in `examples/remote_endpoints/`.

Only one remote/cloud endpoint may be active at a time. When one is active, starting a local profile is refused until the remote endpoint is disabled. API-key **values are never stored in endpoint JSON**; configs contain only an environment-variable name such as `PROVIDER_API_KEY`.

### Web/control authentication

Loopback-only use is authentication-optional. Any non-loopback bind is refused unless authentication is configured.

Environment variables:

- `INFERENCEDECK_USER` — login name, default `admin`.
- `INFERENCEDECK_TOKEN` — shared password/token.
- `INFERENCEDECK_TOKEN_FILE` — file containing the shared password/token.

Browser login creates an in-memory session and an `HttpOnly; SameSite=Strict` cookie. Programmatic clients and tray frontends may send the same token in `X-Auth-Token`.

Do not bind the web/control service to a LAN or tailnet address without setting a token; InferenceDeck will fail closed rather than expose unauthenticated process controls.

### Deliberately not owned

- Bundled `llama.cpp`/`vllm.cpp` binaries or compiled AVX/CUDA builds.
- `llama.cpp`/`vllm.cpp` source-build management. InferenceDeck consumes runtimes rather than owning their compiler/build toolchains.
- Machine-specific EVECOR paths, systemd units, Windows task definitions, credentials, or private endpoint addresses.

## Install

From a checkout (InferenceDeck is not published on PyPI):

```bash
python -m pip install -e .          # core, CLI and web UI
python -m pip install -e ".[tray]"  # also the Windows/macOS tray (pystray, Pillow)
```

## CLI

```bash
inferencedeck inventory --pretty
inferencedeck profiles --pretty
```

or:

```bash
python -m inferencedeck inventory --pretty
```

Commands (all print JSON; with no command, `inventory` runs):

| Command | What it does |
|---|---|
| `inventory` | Runtimes, GGUF models and profiles. |
| `profiles` | Profiles from `models.json` resolved against discovered models. |
| `resolved-inventory` | Inventory plus resolved profile matches. |
| `prepare MODE` | Build the `llama-server` command for a profile without launching it. |
| `servers` | Servers started by InferenceDeck. |
| `stop --server-id ID` / `stop --mode MODE` | Stop a tracked server. |
| `logs SERVER_ID [--lines N]` | Read a tracked server's log. |

Discovery commands accept `--project-root`, `--model-dir` (repeatable), `--max-files`
and `--no-manifest`.

## Configuration

| Item | Location |
|---|---|
| Settings (`config.json`) | Config dir: `%APPDATA%\inferencedeck` on Windows, `$XDG_CONFIG_HOME/inferencedeck` or `~/.config/inferencedeck` elsewhere. Override with `LCC_CONFIG_DIR`. |
| Remote/cloud endpoints | `remote_endpoints/*.json` in the config dir. |
| Logs, state, generated launch scripts | Cache dir: `%LOCALAPPDATA%\inferencedeck`, `$XDG_CACHE_HOME/inferencedeck` or `~/.cache/inferencedeck`. Override with `LCC_CACHE_DIR` (and `LCC_LAUNCH_SCRIPTS_DIR` for scripts). |
| Profiles (`models.json`) | The project root: `--project-root`, else the nearest parent of the working directory containing `models.json`, `llama-server`, `switch-model.ps1` or `pyproject.toml`. |

`config.json` keys include `model_dirs`, `runtime_dirs`, `llama_server_path`,
`llama_runtime`, `llama_fit_params_path`, `extra_llama_args`, `vllm_cpp_server_path`,
`extra_vllm_cpp_args`, `default_host` and `default_port` (see `inferencedeck/config.py` for the full list and defaults).

GGUF models are scanned in `model_dirs`, the `LCC_MODEL_DIRS`, `LLAMA_MODELS_DIR` and
`LLAMA_CPP_MODEL_DIRS` path lists, `LLAMA_CPP_HOME/models`, `models/` under the project
root and working directory, common home folders (`~/models`, `~/llms`, …), LM Studio's
model folders and the Hugging Face cache (`HF_HOME`). A whole drive is never scanned.

## Choosing a llama.cpp build

InferenceDeck picks the llama-server build this CPU can run:

1. A pinned build (`llama_runtime` in config, or the web UI's Runtime menu), if it is compatible.
2. The standard build. Standard builds need AVX2.
3. An AVX1 compatibility build with CUDA, when an NVIDIA GPU is present.
4. A CPU-only AVX1 compatibility build.
5. Otherwise it refuses to start and says why each build was rejected.

So AVX2 machines use the normal llama.cpp build, and AVX-only CPUs (e.g. older
Xeons) fall back to a CUDA AVX1 build. It never launches a build that needs an
instruction set the CPU lacks, even if pinned. `llama-fit-params` and `llama-cli`
are taken from the same build folder as the chosen server.

Builds are found at `llama_server_path` or `LLAMA_SERVER`/`LLAMA_SERVER_BIN` (treated as
pins), under `runtime_dirs`, `LLAMA_CPP_HOME`, the project root and working directory
(including `bin`, `build*/bin` and Visual Studio `Release` folders), and on `PATH`. Each build's requirements come from, in order:

- an `inferencedeck-runtime.json` next to the binary:
  ```json
  {"variant": "cuda-avx1", "label": "CUDA AVX1 build", "cpu": {"requires": ["avx", "f16c"]}, "gpu": {"backend": "cuda"}}
  ```
- the build's `CMakeCache.txt` (`GGML_AVX`, `GGML_AVX2`, `GGML_FMA`, `GGML_F16C`, `GGML_CUDA`);
- otherwise it is assumed to be a standard build that needs AVX2.

## Running a profile on vllm.cpp

[vllm.cpp](https://github.com/mudler/vllm.cpp) is a standalone C++ engine (no Python)
with vLLM's serving core: continuous batching, a paged KV cache and prefix caching.
It loads the same GGUF files as llama.cpp, plus Safetensors. It is a second backend
alongside llama.cpp, not a replacement, and it is still alpha: its CLI can change
between releases.

A profile runs on vllm.cpp when its `recommended_params` set `"runtime": "vllm.cpp"`.
The runtime is part of the profile and can't be switched over the control API.

```json
{"mode": "qwen-vllm", "name": "Qwen3 8B (vllm.cpp)",
 "recommended_params": {"runtime": "vllm.cpp", "ctx_size": 32768, "max_num_seqs": 8}}
```

`vllm-server` is found at `vllm_cpp_server_path` in config, `VLLM_CPP_SERVER` /
`VLLM_CPP_SERVER_BIN`, under `VLLM_CPP_HOME` or `runtime_dirs` (`bin/` of a release
archive or `build/examples/` of a source build), or on `PATH`.

| Profile param | `vllm-server` flag |
|---|---|
| `ctx_size` | `--max-model-len`, and `--num-blocks` sized to hold one sequence that long |
| `num_blocks` / `kv_cache_memory_mib` / `block_size` | `--num-blocks` / `--kv-cache-memory` / `--block-size` |
| `max_num_seqs`, `max_num_batched_tokens` | `--max-num-seqs`, `--max-num-batched-tokens` |
| `kv_cache_dtype` | `--kv-cache-dtype` (e.g. `fp8`) |
| `reasoning` | `--enable-thinking` / `--no-enable-thinking` (unset: the chat template decides) |
| `enable_prefix_caching` | `--enable-prefix-caching` / `--no-enable-prefix-caching` |
| `speculative_config` | `--speculative-config` (JSON) |
| `tool_call_parser`, `reasoning_parser`, `scheduling_policy`, `generation_config`, `mmproj` | the flag of the same name |

llama.cpp-only settings (`gpu_layers`, `threads`, `cache_type_k`, …) and sampling
values (vllm-server takes those per request) produce a warning rather than a flag.
Start, Stop, Pause, Release GPU, Restart, the Context presets and Benchmark work the
same as for llama.cpp; Start waits up to 180 s for readiness. Generated launch scripts
call `vllm-server --model …` for these profiles. Fit needs `llama-fit-params` and
stays llama.cpp-only.

## Local control API and web UI

```bash
inferencedeck-web --host 127.0.0.1 --port 8716
```

This one process serves the web UI and the control API (`/api/*`) and is the only
process that changes tracked server state. The Windows and Linux trays talk to it
at `INFERENCEDECK_URL` (default `http://127.0.0.1:8716`).

Server controls (the same in the web UI and both trays):

| Control | What it does |
|---|---|
| **Pause / Resume** | Freezes the server process. The model stays loaded, so VRAM is **not** freed. |
| **Release GPU / Restore** | Stops the server (freeing VRAM) and keeps its profile and settings; Restore starts it again. |
| **Reload & restart** | Stops and starts the server with the same settings. |
| **Context 8K–128K** | Restarts the server (or restores a released one) at that context size. |
| **Stop** | Stops the server. On a released server it forgets the saved settings. |

For authenticated LAN/tailnet use:

```bash
export INFERENCEDECK_USER=admin
export INFERENCEDECK_TOKEN='use-a-secret-from-your-secret-manager'
inferencedeck-web --host 0.0.0.0 --port 8716
```

The example above is illustrative; do not commit the token to the repository or a config file.

### Telemetry

The control process collects telemetry locally and never sends it anywhere. Two
endpoints serve it, behind the same authentication as the rest of the API:

| Endpoint | Format | Contents |
|---|---|---|
| `GET /api/telemetry` | JSON | System CPU, load and memory; live NVIDIA GPU utilisation, VRAM, temperature, power and clock; per-server state, uptime, startup time, context size, resident memory, CPU time and GPU memory; per-server requests and tokens (see below); the latest benchmark per profile; lifecycle counters and the most recent lifecycle events |
| `GET /metrics` | Prometheus text | The same readings as `inferencedeck_*` metrics, one series per profile |

Lifecycle events are `server.started`, `server.ready`, `server.start_failed`,
`server.stopped`, `server.stop_failed`, `server.suspended`, `server.resumed`,
`server.released`, `server.restored` and `benchmark.completed`. Their counters live in memory and reset
when the control process restarts, as Prometheus counters do. Readings are cached
for two seconds so a UI poll and a scrape share one `nvidia-smi` call. GPU readings
need `nvidia-smi`, and per-process memory and CPU time need Linux `/proc`. On other
platforms those fields are left out. They are never reported as zero.

#### Requests and tokens

InferenceDeck doesn't sit between clients and the inference server, so it reads
request and token counts from the server itself. llama-server is launched with
`--metrics` (set `"metrics": false` in a profile to turn this off), and each live
server's `/metrics` counters are read with a one-second timeout. Paused servers
are skipped. The readings for each server are:

| Field | Meaning |
|---|---|
| `requests_active`, `requests_deferred` | Requests being processed, and waiting for a free slot |
| `prompt_tokens_total`, `tokens_generated_total` | Token counters since the server started |
| `tokens_per_second` | Generation speed, over the time the server spent generating since the previous reading. Absent while idle, never zero |
| `prompt_tokens_per_second` | Prompt-processing speed, measured the same way |
| `throughput_tokens_per_second` | Tokens generated per wall-clock second since the previous reading (zero while idle) |
| `kv_cache_usage_percent` | Share of the KV cache in use, on llama.cpp builds that report it |

Rates are worked out from how InferenceDeck's own readings of the counters change,
so another scraper doesn't change them. llama-server resets its
`llamacpp:prompt_tokens_seconds` and `llamacpp:predicted_tokens_seconds` gauges on
every scrape, though, so a separate Prometheus job scraping llama-server directly
sees those two gauges cover shorter windows. The counters it scrapes are unaffected.
vllm.cpp servers are read the same way if they serve vLLM-style `vllm:*` metrics.

#### History and graphs

`inferencedeck-web` also samples the snapshot every 5 seconds and keeps it for the
**Telemetry** card in the web UI. That card has one chart per measure: GPU
utilisation, VRAM, temperature, power and clock; GPU memory, generation speed,
prompt-processing speed, active requests and KV cache per server; CPU and RAM. Hovering one chart moves a crosshair across all of them. Lifecycle events
are marked on the charts, with failed starts and stops in red. A data table
under the charts lists the latest, average and peak value of every series.

The last hour is kept at full resolution in memory. One-minute averages are
written to one JSON-lines file per UTC day under the cache directory's
`telemetry/` folder, so the 6h, 24h and 7d views survive a restart. Lifecycle
events go into the same files. Gaps in a chart mean the control process wasn't
running. The same data is available as `GET /api/telemetry/history?range=15m|1h|6h|24h|7d`.

| Setting (`config.json`) | Default | Meaning |
|---|---|---|
| `telemetry_sample_seconds` | `5` | Seconds between samples; `0` turns history off |
| `telemetry_retention_days` | `7` | Days of one-minute averages kept on disk |

#### Prometheus

Scrape `/metrics` with Prometheus (the token goes in a bearer header, which Prometheus sends natively):

```yaml
scrape_configs:
  - job_name: inferencedeck
    static_configs:
      - targets: ["127.0.0.1:8716"]
    authorization:
      credentials_file: /etc/prometheus/inferencedeck-token   # omit when auth is off
```

#### OpenTelemetry

`inferencedeck-web` can also push telemetry to an OpenTelemetry Collector, or any
OTLP/HTTP receiver. It is off unless an endpoint is set, and it is the only
telemetry that leaves the machine. It speaks OTLP over HTTP with JSON encoding
(the collector's `otlp` receiver on port 4318), with no extra dependencies.

| Signal | What is sent |
|---|---|
| Metrics | The same readings as `/metrics`, every `otlp_export_seconds`. Gauges stay gauges and counters become cumulative sums |
| Logs | One record per lifecycle event. Failed starts and stops are `WARN` |
| Traces | An `inferencedeck.server.startup` span from launch until ready (or until it gave up, with error status), and an `inferencedeck.benchmark` span per benchmark run |

Turn it on with `otlp_endpoint` in `config.json`, or the standard environment variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=Bearer%20...'   # credentials belong here, not in config.json
export OTEL_SERVICE_NAME=thanatos                                # default: inferencedeck
export OTEL_RESOURCE_ATTRIBUTES=deployment.environment=home
inferencedeck-web
```

| Setting (`config.json`) | Default | Meaning |
|---|---|---|
| `otlp_endpoint` | `""` | OTLP/HTTP base URL; `/v1/metrics`, `/v1/logs` and `/v1/traces` are appended |
| `otlp_export_seconds` | `15` | Seconds between exports |

A collector that is down doesn't affect InferenceDeck. Events are queued (up to
2,000) and retried, and a missed metrics round is replaced by the next one.
`GET /api/telemetry` reports the exporter's state under `otlp`: last export time,
last error, and counts exported and dropped.

## Development

```bash
python -m unittest discover -s tests -v
```

CI exercises Python 3.10 and 3.12 on Linux, Windows, and macOS, including the tray's menu logic. A separate job syntax-checks the Linux tray.

## Provenance

InferenceDeck refracts functionality proven in the earlier Llama Control Center, Thanatos Windows tray, and EVECOR Linux tray/web control experiments into one portable core with multiple thin frontends. The donor environments remain historical/reference material rather than runtime dependencies.
