# InferenceDeck

Cross-platform control plane for local AI inference — discover runtimes and GGUF models, detect hardware, estimate fit, manage profiles, benchmark performance, and control local or remote inference endpoints.

## Current scope

InferenceDeck is a clean continuation of the portable core developed in the earlier local **Llama Control Center** project. Historical Windows and Linux control surfaces were used as donor implementations, but machine-specific paths, service files, compiled runtimes, and private endpoint configuration are intentionally excluded.

### Core

- Discover local `llama.cpp`, `vllm.cpp`, MLC LLM, Ollama, LM Studio, vLLM, and MLX runtimes.
- Discover GGUF models from configured and common model locations.
- Detect CPU, GPU, system memory, VRAM, and available acceleration backends.
- Estimate model fit and performance, with `llama-fit-params` integration when available.
- Resolve and manage portable model profiles.
- Prepare `llama-server`, `vllm-server` (vllm.cpp) or `mlc_llm serve` (MLC LLM) launch commands and manage servers started by InferenceDeck.
- Pause/resume tracked servers without losing process state, or release the GPU (stop the server, keep its settings) and restore it later.
- Benchmark local OpenAI-compatible inference endpoints and retain bounded benchmark history.
- Inspect Hugging Face tooling and runtime update availability.
- Generate portable launch scripts without overwriting hand-written scripts.
- Serve one OpenAI- and Anthropic-compatible inference API in front of whichever local or remote target is active (`inferencedeck-gateway`).

### Frontends

- Local browser control panel (`inferencedeck-web`).
- Python system tray for Windows and macOS (`inferencedeck-tray`, see `frontends/windows/`). No .NET runtime needed.
- Linux GTK/AyatanaAppIndicator tray under `frontends/linux/`.
- All frontends use the same InferenceDeck control API; they do not duplicate model discovery or server-management logic.

### Remote and cloud endpoints

InferenceDeck supports two endpoint lanes, which say who runs the model:

- `remote_host` — a self-hosted runtime on another machine you control, such as `llama.cpp` on a GPU box reached over a LAN or tailnet.
- `true_cloud` — a hosted API such as OpenRouter, where the request leaves your infrastructure.

How requests reach the endpoint is a separate, optional `transport` field: `tailscale`, `lan`, or `https`. Tailscale is a transport, not a provider. When `transport` is omitted, it is inferred from `baseUrl`: `*.ts.net` names and `100.64.0.0/10` addresses count as Tailscale, and `true_cloud` endpoints default to HTTPS. The optional `host` field names the machine; it defaults to the hostname in `baseUrl`. The tray and web UI label each endpoint from these fields, for example `Qwen3-32B (Thanatos · Tailscale · Self-hosted)` or `Claude Sonnet (OpenRouter · Cloud)`.

Endpoint definitions live under the per-user InferenceDeck configuration directory in `remote_endpoints/*.json`. Generic examples are provided in `examples/remote_endpoints/`.

Only one remote/cloud endpoint may be active at a time. When one is active, starting a local profile is refused until the remote endpoint is disabled. API-key **values are never stored in endpoint JSON**; configs contain only an environment-variable name such as `PROVIDER_API_KEY`. `apiKeyEnv` is required for `true_cloud` endpoints and optional for `remote_host` endpoints, so a self-hosted server that does not check keys, such as `llama-server` without `--api-key`, needs no dummy variable. If a `remote_host` config does name `apiKeyEnv`, that variable must be set before the endpoint can be enabled.

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
| `config validate [--strict] [--json]` | Check `config.json`, `models.json` and remote endpoint files without starting anything (see below). Exits 1 on errors. |
| `config schema config\|profiles\|endpoint` | Print the JSON Schema for a configuration file. |
| `updates [--channel stable\|prerelease] [--refresh]` | Compare each installed runtime's version with its latest upstream release (see below). |

Discovery commands accept `--project-root`, `--model-dir` (repeatable), `--max-files`
and `--no-manifest`.

### Update checks

`inferencedeck updates` compares installed versions with the latest GitHub release
for llama.cpp, Ollama, vLLM, vllm.cpp, MLC LLM and MLX. It never downloads or
replaces anything, and caches results for an hour (`--refresh` skips the cache). The
channel defaults to `update_channel` in config.

- **vllm.cpp** is read from the release archive's `VERSION` file, else from
  `vllm-server --version`, else from a running server's `/version`. A `+cuda`-style
  backend suffix is ignored when comparing.
- **MLC LLM** is read from the installed `mlc-llm*` wheel in the Python environment
  that runs `mlc_llm`. Nightly builds (`0.26.dev94`) compare against release tags as
  PEP 440 orders them. A source checkout that still reports `0.1.dev0` has no real
  version, so it isn't checked.
- A project that tags versions without publishing GitHub releases is checked against
  its highest matching tag instead.

The check isn't shown in the web UI or trays yet.

## Configuration

| Item | Location |
|---|---|
| Settings (`config.json`) | Config dir: `%APPDATA%\inferencedeck` on Windows, `$XDG_CONFIG_HOME/inferencedeck` or `~/.config/inferencedeck` elsewhere. Override with `LCC_CONFIG_DIR`. |
| Remote/cloud endpoints | `remote_endpoints/*.json` in the config dir. |
| Logs, state, generated launch scripts | Cache dir: `%LOCALAPPDATA%\inferencedeck`, `$XDG_CACHE_HOME/inferencedeck` or `~/.cache/inferencedeck`. Override with `LCC_CACHE_DIR` (and `LCC_LAUNCH_SCRIPTS_DIR` for scripts). |
| Profiles (`models.json`) | The project root: `--project-root`, else the nearest parent of the working directory containing `models.json`, `llama-server`, `switch-model.ps1` or `pyproject.toml`. |

`config.json` keys include `model_dirs`, `runtime_dirs`, `llama_server_path`,
`llama_runtime`, `llama_fit_params_path`, `extra_llama_args`, `vllm_cpp_server_path`,
`extra_vllm_cpp_args`, `mlc_llm_path`, `extra_mlc_llm_args`, `default_host`, `default_port`, `idle_release_seconds` and `concurrent_vram_check` (see `inferencedeck/config.py` for the full list and defaults).

GGUF models are scanned in `model_dirs`, the `LCC_MODEL_DIRS`, `LLAMA_MODELS_DIR` and
`LLAMA_CPP_MODEL_DIRS` path lists, `LLAMA_CPP_HOME/models`, `models/` under the project
root and working directory, common home folders (`~/models`, `~/llms`, …), LM Studio's
model folders and the Hugging Face cache (`HF_HOME`). A whole drive is never scanned.

Runtime discovery also checks for already-running servers at `LLAMA_SERVER_URL` (or
`LLAMA_SERVER_HOST`/`LLAMA_SERVER_PORT`), `OLLAMA_HOST`, `LMSTUDIO_HOST`, `VLLM_HOST` and
`VLLM_CPP_SERVER_URL`. `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) is sent with Hugging Face
metadata requests when set.


### Checking the configuration

`inferencedeck config validate` (or `inferencedeck-web --check-config`) checks the three
files that shape what InferenceDeck does, without launching servers, probing hardware or
opening ports:

- `config.json`: key types and allowed values, unknown keys (a typo such as
  `idel_release_seconds` is otherwise silently ignored), and configured paths that don't
  exist. A `config.json` that isn't valid JSON is an error, since InferenceDeck then runs
  on defaults.
- `models.json`: each profile has a unique `mode`, typed `recommended_params` (for
  example `ctx_size` must be an integer and `flash_attn` a boolean), unknown params with a
  "did you mean" hint, and runtimes that are detected but cannot be launched.
- `remote_endpoints/*.json`: the same rules the endpoint loader enforces, plus unknown
  keys; an `apiKey` stored in the file is an error.

Errors mean a file won't work as written; warnings mean it will, but probably not as
meant. `inferencedeck-web` prints any problems when it starts, the web UI shows them in the
Status card, and `GET /api/config/check` returns them as JSON. `--strict` makes warnings
fail the command too, for CI.

The schemas are also in `schemas/` (`config.schema.json`, `models.schema.json`,
`remote-endpoint.schema.json`). Point an editor at one for completion and inline checks,
e.g. `"$schema": "https://raw.githubusercontent.com/Deep-Sixed/InferenceDeck/main/schemas/models.schema.json"`
in `models.json`.
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
archive or `build/examples/` of a source build), or on `PATH`. Discovery also probes a
running vllm.cpp server at `VLLM_CPP_SERVER_URL` (default `http://127.0.0.1:8000`, the
same port vLLM's probe uses).

| Profile param | `vllm-server` flag |
|---|---|
| `ctx_size` | `--max-model-len`, and `--num-blocks` sized to hold one sequence that long |
| `num_blocks` / `kv_cache_memory_mib` / `block_size` | `--num-blocks` / `--kv-cache-memory` / `--block-size` |
| `max_num_seqs`, `max_num_batched_tokens` | `--max-num-seqs`, `--max-num-batched-tokens` |
| `kv_cache_dtype` | `--kv-cache-dtype` (e.g. `fp8`) |
| `reasoning` | `--enable-thinking` / `--no-enable-thinking` (unset: the chat template decides) |
| `enable_prefix_caching` | `--enable-prefix-caching` / `--no-enable-prefix-caching` |
| `speculative_config` | `--speculative-config` (JSON) |
| `tool_call_parser`, `reasoning_parser`, `scheduling_policy`, `generation_config`, `tokenizer_config`, `mmproj` | the flag of the same name |

llama.cpp-only settings (`gpu_layers`, `threads`, `cache_type_k`, …) and sampling
values (vllm-server takes those per request) produce a warning rather than a flag.
Start, Stop, Pause, Release GPU, Restart, the Context presets and Benchmark work the
same as for llama.cpp; Start waits up to 180 s for readiness. Generated launch scripts
call `vllm-server --model …` for these profiles. Fit needs `llama-fit-params` and
stays llama.cpp-only.

## Running a profile on MLC LLM

[MLC LLM](https://github.com/mlc-ai/mlc-llm) compiles models ahead of time with Apache
TVM and serves them on CUDA, Metal, Vulkan, ROCm or OpenCL. It does **not** load GGUF:
it serves MLC weight folders (those with an `mlc-chat-config.json`, such as the
`mlc-ai/*-MLC` repos on Hugging Face). So an MLC profile names its model with
`mlc_model` instead of being matched against discovered GGUF files:

```json
{"mode": "qwen-mlc", "name": "Qwen3 8B (MLC LLM)",
 "recommended_params": {"runtime": "mlc-llm", "mlc_model": "HF://mlc-ai/Qwen3-8B-q4f16_1-MLC",
                        "ctx_size": 16384, "mlc_mode": "server"}}
```

`mlc_model` is a local MLC folder or an `HF://org/repo` id, which `mlc_llm serve`
downloads into its own cache on first start. InferenceDeck runs the `mlc_llm` command
from `mlc_llm_path` in config, `MLC_LLM_BIN` or `PATH`, or `python -m mlc_llm` when
the package is installed in its own Python environment.

| Profile param | `mlc_llm serve` argument |
|---|---|
| `mlc_model` | the model (positional) |
| `mlc_mode` | `--mode` (`local`, `interactive` or `server`) |
| `device` | `--device` (e.g. `cuda:0`, `metal`, `vulkan`; default `auto`) |
| `model_lib` | `--model-lib` (otherwise MLC JIT-compiles one) |
| `enable_prefix_caching` | `--prefix-cache-mode radix` / `disable` |
| `ctx_size`, `max_num_seqs`, `max_total_seq_length`, `prefill_chunk_size`, `gpu_memory_utilization`, `tensor_parallel_shards`, `sliding_window_size` | `--overrides` (`context_window_size`, `max_num_sequence`, … ) |

As with vllm.cpp, llama.cpp-only settings and sampling values produce a warning.
Start waits up to 600 s, since the first start may download weights and compile a
model library. Generated launch scripts call `mlc_llm serve $model …`; Fit stays
llama.cpp-only.

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

### Idle auto-release

InferenceDeck can release a server's GPU (the same as **Release GPU**) once it has
gone a set time without requests; **Restore** starts it again with the same settings.
It is off by default. Set a default for all servers with `idle_release_seconds` in the
app config, or per server with the **Auto-release** buttons in the web UI (`POST
/api/idle` with `server_id` and `seconds`, or `null` to use the default). A profile
param or launch override named `idle_release_seconds` sets it at start.

InferenceDeck is not in the request path, so `inferencedeck-web` polls each server's
llama-server `/slots` every 15 seconds. A slot that is processing, or one that took a
new task since the last poll, counts as activity. A server whose `/slots` cannot be
read (started with `--no-slots` or `--api-key`, not responding, or a vllm.cpp server,
which has no `/slots`) is never auto-released. The idle clock restarts when `inferencedeck-web` restarts.

### Model capabilities

Profiles (`GET /api/profiles`) and tracked servers carry a `capabilities` object:
`input` (`text`, `image`, `audio`), `output` (`text`, `embedding` or `score`), `tools`,
`embedding`, `reranker` and `context_max` (the context the model was trained for), with
`sources` saying where each came from. Filter profiles with
`GET /api/profiles?capability=tools` (or `image`, `audio`, `embedding`, `reranker`).

- **GGUF header** (read once and cached, never while listing): a tool-calling chat
  template, the trained context length, and the pooling type, which marks embedding
  and reranking models. Until a model has been started or fitted once, the profile list
  shows these as unknown.
- **The launch:** image/audio input only when a projector is passed. Set
  `"vision": true` on the profile (or as a launch override) to pass the `mmproj-*.gguf`
  found next to the model as `--mmproj`, or name one with `"mmproj"`. The web UI notes
  when a projector is found but not enabled. The projector's size is included in the GPU
  memory check. `"reranking": true` adds `--reranking` for reranker models (`"embedding":
  true` already adds `--embedding`).
- **The profile:** a `capabilities` object in the profile's params overrides any
  field, e.g. `{"capabilities": {"tools": false, "context_max": 32768}}`.
- **The running server:** once llama-server is ready, what its `/props` reports
  (modalities, tool support, per-request context) replaces the guesses.

Generated launch scripts now use the same launch params as Start, so they also pick up the
profile's sampling preset and projector.

### Live logs

Each tracked server has a **Logs** button in the web UI that opens a live view of its
log. Behind it is `GET /api/logs/stream?server_id=…` (server-sent events; add
`&stream=stderr` or `stdout` for one file, `&history=` bytes of backlog, default 64 KiB).
It sends recent history, then new lines as llama-server writes them.

Servers write straight to their log files under the cache dir, so the files stay the
complete record and a slow browser can never stall a server. The stream is what's
bounded: history is capped, a viewer that falls more than 1 MiB behind skips ahead
(and is told how much it missed), at most 8 streams run at once, and the web view
keeps the last 2000 lines. A stream ends when its server is no longer tracked; a
restart gives the server a new id, so reopen **Logs** after one. `GET /api/logs` (the
last N lines) now reads only the end of the file instead of loading all of it.

### Starting a server next to running ones

Before a llama.cpp server starts (Start, Restore, Reload & restart, Benchmark), InferenceDeck
checks that it fits in GPU memory next to the servers already running. It estimates the new
server's VRAM with the same estimator as the fit badges, and compares it with:

- **free VRAM right now**, from `nvidia-smi` (or available memory on Apple silicon), which
  already counts running servers; or, when that isn't available,
- **total VRAM minus the estimates of the tracked servers that are running** (paused ones
  included, since they keep their VRAM; released ones are not counted).

If the model would fit on its own but not alongside what is running, the start is refused
(HTTP 409, `"reason": "vram_conflict"`) with a `vram_plan` naming the fewest servers to
release, biggest first. The web UI then offers to release them and continue (Restore brings
them back) or to start anyway. Over the API, send `"release_conflicts": true` or
`"force": true` with `/api/start`, `/api/restore` or `/api/restart`. `POST /api/plan` with a
`mode` reports the plan without starting anything.

A tight fit, or a model too big for the GPU even alone, starts with a warning as before;
the Fit tools are the place to shrink it. Set `concurrent_vram_check` in the app config to
`"warn"` to never refuse, or `"off"` to skip the check. Only the primary GPU is checked, and
vllm.cpp launches are not checked.

### Request defaults and sampling presets

A profile can set what requests get when they don't choose for themselves:

```json
{
  "sampling_preset": "coding",
  "temperature": 0.3,
  "n_predict": 2048,
  "jinja": true,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

- `sampling_preset` is one of `coding`, `factual`, `balanced` or `creative`
  (`GET /api/sampling` lists their values). A profile's own values sit on top of
  its preset, so the example above is the coding preset with `temperature` 0.3.
- A preset picked at launch (the web UI's preset menu next to **Start**, or
  `"overrides": {"sampling_preset": "creative"}` on `POST /api/start`) replaces the
  profile's sampling values; other explicit overrides still win over it. `"none"`
  drops the profile's preset.
- `chat_template_kwargs` becomes llama-server's `--chat-template-kwargs`, for switches
  such as `enable_thinking` or `reasoning_effort`. It is read by the Jinja chat
  template, so pair it with `"jinja": true`. It can only be set in the profile, not
  over the API.

These become llama-server launch flags, so they are **defaults**: a request that sends
its own `temperature`, `max_tokens` or `chat_template_kwargs` still gets what it asked
for. The web UI shows each running server's defaults. Forcing a value over the
client's (llama-swap's `setParams`) would need InferenceDeck in the request path, which
it is not. vllm.cpp servers take sampling per request only, so presets there only
produce the usual "not applied" warning.

For authenticated LAN/tailnet use:

```bash
export INFERENCEDECK_USER=admin
export INFERENCEDECK_TOKEN='use-a-secret-from-your-secret-manager'
inferencedeck-web --host 0.0.0.0 --port 8716
```

The example above is illustrative; do not commit the token to the repository or a config file.

## Inference gateway (API mapping)

```bash
inferencedeck-gateway --host 127.0.0.1 --port 8717
```

The gateway gives applications one stable inference API, whatever is serving the model. It sends each request to the current target:

1. the enabled remote/cloud endpoint, if there is one;
2. otherwise the running (not paused) local server.

The app keeps the same URL whether the model is on this machine, on another box over Tailscale, or on OpenRouter.

| Client API | Path |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| Anthropic Messages | `POST /v1/messages` |
| Model list (the current target) | `GET /v1/models` |

Requests are translated through one internal request format, so each API and each engine needs only one adapter. That means N + M adapters rather than one per API/engine pair.

The translation covers:

- messages and system prompts
- images
- tool definitions, tool calls and tool results
- sampling (`temperature`, `top_p`, `top_k`, `min_p`, penalties, seed, stop sequences)
- streaming, finish reasons and token usage
- errors, returned in the caller's own format

For example, an Anthropic SDK can talk to a local `llama.cpp` server. Engine-specific OpenAI fields (such as `repeat_penalty`) are passed through unchanged.

The only engine adapter so far is OpenAI-compatible. It covers `llama.cpp`, `vllm.cpp`, vLLM, LM Studio and OpenRouter. The Anthropic API's `thinking` setting is dropped, and its server tools (such as `web_search`) are rejected.

API keys for remote endpoints are attached by the gateway from `apiKeyEnv`. A client's own key is never forwarded upstream.

The gateway uses the same bind rule as the control API: loopback only, unless `INFERENCEDECK_TOKEN` is set. With a token set, clients send it as their API key (`Authorization: Bearer …` or `x-api-key`), so standard OpenAI and Anthropic SDKs work unchanged.

## Development

```bash
python -m unittest discover -s tests -v
```

CI exercises Python 3.10 and 3.12 on Linux, Windows, and macOS, including the tray's menu logic. A separate job syntax-checks the Linux tray.

## Provenance

InferenceDeck refracts functionality proven in the earlier Llama Control Center, Thanatos Windows tray, and EVECOR Linux tray/web control experiments into one portable core with multiple thin frontends. The donor environments remain historical/reference material rather than runtime dependencies.
