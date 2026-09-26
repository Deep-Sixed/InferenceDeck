# InferenceDeck

Cross-platform control plane for local AI inference — discover runtimes and GGUF models, detect hardware, estimate fit, manage profiles, benchmark performance, and control local or remote inference endpoints.

## Current scope

InferenceDeck is a clean continuation of the portable core developed in the earlier local **Llama Control Center** project. Historical Windows and Linux control surfaces were used as donor implementations, but machine-specific paths, service files, compiled runtimes, and private endpoint configuration are intentionally excluded.

### Core

- Discover local `llama.cpp`, Ollama, LM Studio, vLLM, and MLX runtimes.
- Discover GGUF models from configured and common model locations.
- Detect CPU, GPU, system memory, VRAM, and available acceleration backends.
- Estimate model fit and performance, with `llama-fit-params` integration when available.
- Resolve and manage portable model profiles.
- Prepare `llama-server` launch commands and manage servers started by InferenceDeck.
- Suspend/resume tracked servers to free accelerator resources without losing process state.
- Benchmark local OpenAI-compatible inference endpoints and retain bounded benchmark history.
- Inspect Hugging Face tooling and runtime update availability.
- Generate portable launch scripts without overwriting hand-written scripts.

### Frontends

- Local browser control panel (`inferencedeck-web`).
- Native Windows `.NET 8` system tray under `frontends/windows/`.
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

- Bundled `llama.cpp` binaries or compiled AVX/CUDA builds.
- `llama.cpp` source-build management. InferenceDeck consumes runtimes rather than owning their compiler/build toolchains.
- Machine-specific EVECOR paths, systemd units, Windows task definitions, credentials, or private endpoint addresses.

## Install

```bash
python -m pip install -e .
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

Builds are found under `runtime_dirs`, `LLAMA_CPP_HOME`, the project root (including
`build*/bin`) and `PATH`. Each build's requirements come from, in order:

- an `inferencedeck-runtime.json` next to the binary:
  ```json
  {"variant": "cuda-avx1", "label": "CUDA AVX1 build", "cpu": {"requires": ["avx", "f16c"]}, "gpu": {"backend": "cuda"}}
  ```
- the build's `CMakeCache.txt` (`GGML_AVX`, `GGML_AVX2`, `GGML_FMA`, `GGML_F16C`, `GGML_CUDA`);
- otherwise it is assumed to be a standard build that needs AVX2.

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

## Development

```bash
python -m unittest discover -s tests -v
```

CI exercises Python 3.10 and 3.12 on Linux, Windows, and macOS. Separate jobs build the Windows tray and syntax-check the Linux tray.

## Provenance

InferenceDeck refracts functionality proven in the earlier Llama Control Center, Thanatos Windows tray, and EVECOR Linux tray/web control experiments into one portable core with multiple thin frontends. The donor environments remain historical/reference material rather than runtime dependencies.
