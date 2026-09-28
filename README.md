# InferenceDeck

Cross-platform control plane for local AI inference — discover runtimes and GGUF models, detect hardware, estimate fit, manage profiles, benchmark performance, and control local or remote inference endpoints.

## Current scope

InferenceDeck is a clean continuation of the portable core developed in the earlier local **Llama Control Center** project. Historical Windows and Linux control surfaces were used as donor implementations, but machine-specific paths, service files, compiled runtimes, and private endpoint configuration are intentionally excluded.

### Core

- Discover local `llama.cpp`, `vllm.cpp`, MLC LLM, KoboldCpp, Ollama, LM Studio, vLLM, and MLX runtimes.
- Discover GGUF models from configured and common model locations.
- Detect CPU, GPU, system memory, VRAM, and available acceleration backends.
- Estimate model fit and performance, with `llama-fit-params` integration when available.
- Resolve and manage portable model profiles.
- Prepare `llama-server`, `vllm-server` (vllm.cpp), `mlc_llm serve` (MLC LLM) or KoboldCpp launch commands and manage servers started by InferenceDeck.
- Pause/resume tracked servers without losing process state, or release the GPU (stop the server, keep its settings) and restore it later.
- Benchmark local OpenAI-compatible inference endpoints and retain bounded benchmark history.
- Inspect Hugging Face tooling and runtime update availability.
- Generate portable launch scripts without overwriting hand-written scripts.
- Serve one OpenAI- and Anthropic-compatible inference API in front of whichever local or remote target is active (`inferencedeck-gateway`), optionally loading the profile each request names.

### Frontends

- Local browser control panel (`inferencedeck-web`).
- Python system tray for Windows and macOS (`inferencedeck-tray`, see `frontends/windows/`). No .NET runtime needed.
- Linux GTK/AyatanaAppIndicator tray under `frontends/linux/`.
- All frontends use the same InferenceDeck control API; they do not duplicate model discovery or server-management logic.

### Remote and cloud endpoints

InferenceDeck supports two endpoint lanes, which say who runs the model:

- `remote_host` — a self-hosted runtime on another machine you control, such as `llama.cpp` on a GPU box reached over a LAN or tailnet. An endpoint whose `provider` is `llamacpp` or `ollama` defaults to this lane when `lane` is omitted.
- `true_cloud` — a hosted API such as OpenRouter, where the request leaves your infrastructure.

How requests reach the endpoint is a separate, optional `transport` field: `tailscale`, `lan`, or `https`. Tailscale is a transport, not a provider. When `transport` is omitted, it is inferred from `baseUrl`: `*.ts.net` names and tailnet addresses (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`) count as Tailscale, `true_cloud` endpoints default to HTTPS, and anything else is left unlabelled. Any other `transport` value makes the endpoint invalid. The optional `host` field names the machine; it defaults to the hostname in `baseUrl`. The tray and web UI label each endpoint from these fields, for example `Qwen3-32B (Thanatos · Tailscale · Self-hosted)` or `Claude Sonnet (OpenRouter · Cloud)`.

Endpoint definitions live under the per-user InferenceDeck configuration directory in `remote_endpoints/*.json`. Generic examples are provided in `examples/remote_endpoints/`.

Only one remote/cloud endpoint may be active at a time. When one is active, starting a local profile is refused until the remote endpoint is disabled. API-key **values are never stored in endpoint JSON**; configs contain only an environment-variable name such as `PROVIDER_API_KEY`. `apiKeyEnv` and `model` are required for `true_cloud` endpoints (the model is pinned, so a client's default model name is never billed to your key; pick another model explicitly with `<endpoint>/<model>` through the gateway). `apiKeyEnv` is optional for `remote_host` endpoints, so a self-hosted server that does not check keys, such as `llama-server` without `--api-key`, needs no dummy variable. If a `remote_host` config does name `apiKeyEnv`, that variable must be set before the endpoint can be enabled.

### Web/control authentication

Loopback-only use is authentication-optional. Any non-loopback bind is refused unless authentication is configured.

Environment variables:

- `INFERENCEDECK_USER` — login name, default `admin`.
- `INFERENCEDECK_TOKEN` — shared password/token.
- `INFERENCEDECK_TOKEN_FILE` — file containing the shared password/token.
- `INFERENCEDECK_TRUSTED_PROXIES` — comma-separated IP addresses or CIDR ranges (`10.0.0.0/8`; `localhost` means loopback) of reverse proxies in front of InferenceDeck. Failed logins are throttled per client address (5 per 5 minutes; IPv6 clients per /64), and at most 50 failures from all clients together per 5 minutes, after which everyone waits until the window clears. Behind a proxy every request comes from the proxy, so list it here and InferenceDeck throttles by the client in `X-Forwarded-For` instead. This applies to the control API and the inference gateway alike. The header is ignored from any other address, so clients can't use it to dodge the throttle. Entries that aren't addresses or ranges (hostnames) are reported at startup and ignored.

Browser login creates an in-memory session and an `HttpOnly; SameSite=Strict` cookie (also `Secure` when served over HTTPS). Programmatic clients and tray frontends may send the same token in `X-Auth-Token`.

Do not bind the web/control service to a LAN or tailnet address without setting a token; InferenceDeck will fail closed rather than expose unauthenticated process controls.

Over plain HTTP the token and session cookie cross the network unencrypted. For a LAN bind, serve HTTPS (a tailnet already encrypts traffic between its devices):

```bash
inferencedeck-web --host 0.0.0.0 --certfile cert.pem --keyfile key.pem
```

The trays verify the certificate, so use one they trust (for example a Tailscale or internal-CA certificate) and point `INFERENCEDECK_URL` at `https://`.

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
| `hf-files REPO` | List the GGUF quants (split shards grouped) and vision projectors in a Hugging Face repo. |
| `pull REPO --quant Q4_K_M` / `pull REPO --pattern GLOB` | Download one quant: every shard, plus the repo's mmproj (`--no-mmproj` to skip). `--dry-run` shows the files first. |
| `updates [--channel stable\|prerelease] [--refresh]` | Compare each installed runtime's version with its latest upstream release (see below). |

Discovery commands accept `--project-root`, `--model-dir` (repeatable), `--max-files`
and `--no-manifest`.

### Update checks

`inferencedeck updates` compares installed versions with the latest GitHub release
for llama.cpp, Ollama, vLLM, vllm.cpp, MLC LLM, KoboldCpp and MLX. It never downloads or
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

The web UI's **Runtime updates** card and both trays' **Runtime updates** menu show
the same results (API: `GET /api/updates` for cached results; `POST /api/updates` with a JSON body skips the cache and asks GitHub again). They check
when they start and then every 30 minutes (web) or hourly (trays); **Check now**
asks GitHub again. Each update links to its GitHub release page.

## Configuration

| Item | Location |
|---|---|
| Settings (`config.json`) | Config dir: `%APPDATA%\inferencedeck` on Windows, `$XDG_CONFIG_HOME/inferencedeck` or `~/.config/inferencedeck` elsewhere. Override with `LCC_CONFIG_DIR`. |
| Remote/cloud endpoints | `remote_endpoints/*.json` in the config dir. |
| Logs, state, generated launch scripts | Cache dir: `%LOCALAPPDATA%\inferencedeck`, `$XDG_CACHE_HOME/inferencedeck` or `~/.cache/inferencedeck`. Override with `LCC_CACHE_DIR` (and `LCC_LAUNCH_SCRIPTS_DIR` for scripts). |
| Profiles (`models.json`) | The project root: `--project-root`, else the nearest parent of the working directory containing `models.json`, `llama-server`, `switch-model.ps1` or `pyproject.toml`. |

`config.json` keys include `model_dirs`, `runtime_dirs`, `llama_server_path`,
`llama_runtime`, `llama_fit_params_path`, `extra_llama_args`, `vllm_cpp_server_path`,
`extra_vllm_cpp_args`, `mlc_llm_path`, `extra_mlc_llm_args`, `koboldcpp_path`,
`extra_koboldcpp_args`, `default_host`, `default_port` and `server_history_limit` (see
`inferencedeck/config.py` for the full list and defaults). Stopped servers stay listed,
with their logs, as history; `server_history_limit` (default 5) sets how many are kept
before the oldest records and their log files are deleted.

GGUF models are scanned in `model_dirs`, the `LCC_MODEL_DIRS`, `LLAMA_MODELS_DIR` and
`LLAMA_CPP_MODEL_DIRS` path lists, `LLAMA_CPP_HOME/models`, `models/` under the project
root and working directory, common home folders (`~/models`, `~/llms`, …), LM Studio's
model folders and the Hugging Face cache (`HF_HOME`). A whole drive is never scanned.

Runtime discovery also checks for already-running servers at `LLAMA_SERVER_URL` (or
`LLAMA_SERVER_HOST`/`LLAMA_SERVER_PORT`), `OLLAMA_HOST`, `LMSTUDIO_HOST`, `VLLM_HOST`,
`VLLM_CPP_SERVER_URL` and `MLC_LLM_SERVER_URL`. `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) is sent with Hugging Face
metadata requests when set.

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

## Downloading models from Hugging Face

`inferencedeck pull` (and `POST /api/hf/download`) fetch exactly one quant of a repo.
Split GGUFs (`-00001-of-00003`) are grouped so all shards come down together, and a
vision projector (`mmproj`, preferring F16) is added when the repo has one. No match,
or more than one, fails with the list of available quants instead of guessing.

```bash
inferencedeck hf-files unsloth/gemma-3-4b-it-GGUF --pretty
inferencedeck pull unsloth/gemma-3-4b-it-GGUF --quant Q4_K_M
inferencedeck pull unsloth/gemma-3-4b-it-GGUF --pattern '*UD-Q4_K_XL*' --dest ~/models
```

Files go to the Hugging Face cache (already scanned for models) unless `--dest` is
given; the control API always uses the cache. The download runs the `hf` CLI
(`huggingface-cli` on older installs, from `pip install huggingface_hub`), which
resumes interrupted downloads and uses `HF_TOKEN` for gated repos.

## Multi-GPU, LoRA and other launch options

Profile `recommended_params` map to `llama-server` flags:

| Param | Flag | Example |
|---|---|---|
| `split_mode` | `--split-mode` | `"layer"`, `"row"`, `"tensor"`, `"none"` |
| `tensor_split` | `--tensor-split` | `[3, 1]` or `"3,1"` |
| `main_gpu` | `--main-gpu` | `0` |
| `rpc_servers` | `--rpc` | `["10.0.0.2:50052"]` |
| `lora` | `--lora` / `--lora-scaled` | `["style.gguf", {"path": "domain.gguf", "scale": 0.5}]` |
| `override_kv` | `--override-kv` (one per entry) | `["tokenizer.ggml.add_bos_token=bool:false"]` |
| `rope_scaling`, `rope_scale`, `rope_freq_base`, `rope_freq_scale` | `--rope-*` | `"yarn"`, `4` |
| `yarn_orig_ctx`, `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_slow`, `yarn_beta_fast` | `--yarn-*` | `32768` |
| `numa` | `--numa` | `true` (distribute), `"isolate"`, `"numactl"` |
| `mmap`, `mlock` | `--load-mode`, or `--no-mmap`/`--mlock` on older builds | `false`, `true` |
| `load_mode` | `--load-mode` (translated for older builds) | `"mmap+mlock"`, `"dio"` |

llama.cpp renames flags between releases, so InferenceDeck reads each `llama-server`'s
`--help` once (cached until the binary changes) and spells renamed flags the way that
build expects: `--load-mode` versus `--no-mmap`/`--mlock`, and `--spec-draft-n-max`/`-n-min`
versus `--draft-max`/`--draft-min` for the `draft_max`/`draft_min` keys. A flag the build
doesn't list is reported as a warning before launch.

Invalid values are left out of the command and reported as warnings. These params pick
hardware and files, so they live in the profile and can't be changed over the control
API. The fit test passes the split to `llama-fit-params` and applies the `-ts`/`-sm`/`-mg`
it suggests.

The memory-fit estimate and Smart Tune size a split across every GPU it uses: all
discrete GPUs on the primary GPU's backend by default (llama.cpp's own default), only
`main_gpu` with `split_mode: "none"`, the listed ones with `device: "CUDA0,CUDA1"`, and
with `tensor_split` the card that fills first bounds the total. Each GPU is charged its
own runtime overhead and keeps its own headroom.

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
the `mlc-llm` package is installed in InferenceDeck's own Python environment. Discovery
also probes a running MLC server at `MLC_LLM_SERVER_URL` (default `http://127.0.0.1:8000`).

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

## Running a profile on KoboldCpp

[KoboldCpp](https://github.com/LostRuins/koboldcpp) is a llama.cpp fork shipped as a
single executable. It loads the same GGUF files, so a KoboldCpp profile is matched
against discovered models exactly like a llama.cpp one; set `"runtime": "koboldcpp"`:

```json
{"mode": "qwen-kobold", "name": "Qwen3 8B (KoboldCpp)",
 "recommended_params": {"runtime": "koboldcpp", "ctx_size": 16384}}
```

InferenceDeck runs the executable from `koboldcpp_path` in config, `KOBOLDCPP_BIN`,
`KOBOLDCPP_HOME` or `runtime_dirs` (the release names `koboldcpp`,
`koboldcpp-linux-x64`, `koboldcpp-mac-arm64`, `koboldcpp.exe`, `koboldcpp_nocuda.exe`,
… are all recognized) or `PATH`, or `koboldcpp.py` under Python for a source checkout.
It always passes `--skiplauncher`, so the Tk launcher never opens.

KoboldCpp picks its own GPU backend (CUDA, Vulkan or CPU, with no-AVX2 and failsafe
modes for older CPUs), thread counts and GPU layers (autofit) when a profile leaves them
unset, so a profile needs nothing beyond `runtime` (its context defaults to 16K). llama.cpp-style
settings map to KoboldCpp's flags:

| Profile param | KoboldCpp flag |
|---|---|
| `ctx_size`, `threads`, `threads_batch` | `--contextsize`, `--threads`, `--blasthreads` |
| `gpu_layers` | `--gpulayers` (`auto` or unset: KoboldCpp's autofit) |
| `acceleration_backend` `cuda`/`rocm`/`vulkan`/`cpu`, `device` | `--usecuda`/`--usevulkan`/`--usecpu`, with the device's GPU index |
| `batch_size` | `--batchsize`, rounded down to one KoboldCpp accepts (16-4096, powers of two) |
| `cache_type_k`/`cache_type_v` | `--quantkv` (one type for both: f16, bf16, q8_0, q5_1, q4_0) |
| `flash_attn: false`, `kv_offload: false`, `mmap: true` | `--noflashattention`, `--lowvram`, `--usemmap` |
| `jinja`, `reasoning` | `--jinja_tools`, `--jinjathink true/false` |
| `draft_model`, `draft_max` | `--draftmodel`, `--draftamount` |
| `tensor_overrides`, `mmproj`, `n_predict` | `--overridetensors`, `--mmproj`, `--defaultgenamt` |

Settings KoboldCpp has no flag for (`ubatch_size`, `cache_reuse`, …) and sampling values
produce a warning. Start waits up to 120 s, since one-file builds unpack themselves
before loading. Generated launch scripts call it with `--model $model …`; update checks
use `LostRuins/koboldcpp` releases; Fit stays llama.cpp-only.

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

Starts run one at a time. Start is refused when the profile already has a running server
(unless the caller asks to stop it first) or when another tracked server is using the same
port.

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

The gateway gives applications one stable inference API, whatever is serving the model. The app keeps the same URL whether the model is on this machine, on another box over Tailscale, or on OpenRouter.

### Routing by model name

The gateway picks a target from the `model` name in each request. It can route to:

- every running (not paused) local server;
- every self-hosted (`remote_host`) endpoint;
- cloud endpoints that opt in with `"routable": true`. Cloud endpoints are left out by default, so no request leaves your machines unless you allow it.

A self-hosted endpoint can opt out with `"routable": false`. A cloud endpoint whose API key is not set is left out.

Names are matched without regard to case:

| Target | Names that route to it |
|---|---|
| Remote endpoint | its `aliases`, its `model`, and its file name (`thanatos.json` → `thanatos`) |
| Local server | its profile name, its server id, and its model file name without `.gguf` |

To use any model on a particular endpoint, write `<endpoint>/<model>`. For example, `openrouter/meta-llama/llama-3.3-70b-instruct` sends `meta-llama/llama-3.3-70b-instruct` to the endpoint in `openrouter.json`.

A request with no model name, or one that matches nothing, goes to the **default target**:

1. the enabled remote/cloud endpoint, if there is one;
2. otherwise the first running local server.

This means clients with a hard-coded model name keep working. An enabled endpoint that cannot be used (for example, its key is missing) returns an error rather than silently sending the request somewhere else.

`GET /v1/models` lists every target the gateway can route to, with its aliases, and marks the default. Replies report the model name the client asked for.

```json
{ "provider": "llamacpp", "lane": "remote_host", "host": "Thanatos",
  "baseUrl": "http://thanatos:8080", "model": "qwen3-32b", "aliases": ["big-qwen"] }
```

| Client API | Path |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| Anthropic Messages | `POST /v1/messages` |
| Model list (every routable name, plus loadable profiles when switching is on) | `GET /v1/models` |
| Health check (no token needed) | `GET /healthz` |

Requests are translated through one internal request format, so each API and each engine needs only one adapter. That means N + M adapters rather than one per API/engine pair.

The translation covers:

- messages and system prompts
- images
- tool definitions, tool calls and tool results
- sampling (`temperature`, `top_p`, `top_k`, `min_p`, penalties, seed, stop sequences)
- streaming, finish reasons and token usage
- errors, returned in the caller's own format

For example, an Anthropic SDK can talk to a local `llama.cpp` server. Engine-specific OpenAI fields (such as `repeat_penalty`) are passed through unchanged.

There are three engine adapters:

- **OpenAI-compatible**, for `llama.cpp`, `vllm.cpp`, vLLM, LM Studio and OpenRouter.
- **Native Ollama** (`/api/chat`), used for endpoints with `"provider": "ollama"`. Its `baseUrl` is the server root, e.g. `http://thanatos:11434`; a URL ending in `/v1` or `/api` also works.
  - Sampling settings become Ollama `options`, with `max_tokens` sent as `num_predict`.
  - Other engine fields go into `options` (e.g. `num_ctx`, `repeat_penalty`), except `keep_alive` and `think`, which go at the top level. OpenAI-only fields such as `parallel_tool_calls` are dropped.
  - `response_format` becomes Ollama's `format` (JSON mode or a JSON schema).
  - Images must be inline (base64); image URLs are refused with a 400.
  - Ollama has no `tool_choice`. `none` is honoured by not offering the tools; a forced or required choice is left to the model.
- **Anthropic Messages API** (`/v1/messages`), used for endpoints with `"provider": "anthropic"`. Its `baseUrl` is the API root, `https://api.anthropic.com`; a URL ending in `/v1` or `/v1/messages` also works. The key from `apiKeyEnv` is sent as `x-api-key`. See `examples/remote_endpoints/anthropic.example.json`.
  - System messages are combined into the top-level `system` prompt.
  - Tool results become `tool_result` blocks, and results for parallel calls share one user turn. Tool-call ids are rewritten to the characters the API allows.
  - `max_tokens` is required by the API. When the client leaves it out, the gateway sends 16000, or 64000 for a streamed request.
  - Only `temperature`, `top_p`, `top_k` and stop sequences are forwarded, because the API rejects unknown fields. `min_p`, penalties, `seed` and engine-specific fields are dropped. Newer Claude models also reject `temperature`/`top_p`/`top_k`; that error reaches the client unchanged.
  - A JSON-schema `response_format` becomes `output_config.format`. JSON mode without a schema has no equivalent and is ignored.
  - `required` becomes `tool_choice` `any`, and a named tool becomes `tool`. Some newer models reject forced tool use; that error also reaches the client unchanged.
  - Thinking blocks in replies are not passed on. Prompt caching counts toward the reported input tokens. A 529 (overloaded) reaches the client as a 503.

The Anthropic API's `thinking` setting is dropped, and its server tools (such as `web_search`) are rejected.

API keys for remote endpoints are attached by the gateway from `apiKeyEnv`. A client's own key is never forwarded upstream.

### Switching models on demand

```bash
inferencedeck-gateway --switch-models   # or "gateway_model_switching": true in config.json
```

With switching on, a request can also name a launchable profile that isn't running, by
its mode, display name or `alias`, and the gateway loads it, as Ollama does. Names that
already route somewhere (a running server, a routable endpoint) are used as they are,
without switching. A client can move between local models just by changing `model`:

1. every other running local server is released (stopped; its settings are kept, so
   Restore in the web UI or tray brings it back);
2. the named profile is resumed if paused, restored if released, or started;
3. the request is answered once the server reports ready.

Only one model is loaded at a time, since profiles usually share the GPU and the
default port. Switches are serialized: concurrent requests for the same model load it
once. Before releasing a server, the gateway waits up to two minutes for requests it
is still answering from that server, including streams. It can't see clients that
talk to `llama-server` directly.

The gateway does not start processes itself. Like the trays, it asks the
`inferencedeck-web` control API at `INFERENCEDECK_URL` (default
`http://127.0.0.1:8716`, with `INFERENCEDECK_TOKEN` when set), so that process remains
the only owner of server state. If the control API can't be reached, a switch fails
with a 503 that says so. A model name that matches no profile is not an error: it goes
to the default target as before. While a remote endpoint is enabled, local models are
not loaded (local starts are refused then); requests that name one go to the default
target. `GET /v1/models` adds each loadable profile with `"loaded": false`.

The gateway uses the same bind rule as the control API: loopback only, unless a token is set (`INFERENCEDECK_TOKEN` or `INFERENCEDECK_TOKEN_FILE`). With a token set, clients send it as their API key (`Authorization: Bearer …` or `x-api-key`, or `X-Auth-Token`), so standard OpenAI and Anthropic SDKs work unchanged. Failed tokens are throttled per client like web logins (5 per 5 minutes), and `INFERENCEDECK_TRUSTED_PROXIES` applies here too, so behind a reverse proxy each client keeps its own throttle. POST requests must be sent as `Content-Type: application/json`, as the OpenAI and Anthropic SDKs do; anything else gets 415, which stops a web page you visit from quietly using the gateway.

The gateway serves plain HTTP only; it has no `--certfile` option. For use across a LAN, reach it over a tailnet or put it behind a TLS reverse proxy.

## Development

```bash
python -m unittest discover -s tests -v
```

CI exercises Python 3.10 and 3.12 on Linux, Windows, and macOS, including the tray's menu logic. A separate job syntax-checks the Linux tray.

## Provenance

InferenceDeck refracts functionality proven in the earlier Llama Control Center, Thanatos Windows tray, and EVECOR Linux tray/web control experiments into one portable core with multiple thin frontends. The donor environments remain historical/reference material rather than runtime dependencies.
