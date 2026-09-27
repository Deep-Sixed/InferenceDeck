# InferenceDeck

Cross-platform control plane for local AI inference — discover runtimes and GGUF models, detect hardware, estimate fit, manage profiles, benchmark performance, and launch local model servers.

## Current scope

InferenceDeck begins with the portable core recovered from the earlier Llama Control Center project. The first release intentionally excludes the historical web dashboard and machine-specific integrations so the runtime, model, hardware, fit, benchmarking, profile, and server-management contracts can stabilize independently.

### Included

- Discover local `llama.cpp`, Ollama, LM Studio, vLLM, and MLX runtimes.
- Discover GGUF models from configured and common model locations.
- Detect CPU, GPU, system memory, VRAM, and available acceleration backends.
- Estimate model fit and performance, with `llama-fit-params` integration when available.
- Resolve and manage portable model profiles.
- Prepare `llama-server` launch commands and manage servers started by InferenceDeck.
- Benchmark local OpenAI-compatible inference endpoints.
- Inspect Hugging Face tooling and runtime update availability.
- Generate portable launch scripts without overwriting hand-written scripts.

### Not included yet

- The historical Llama Control Center FastAPI/web dashboard.
- EVECOR-specific paths, services, or configuration.
- Bundled `llama.cpp` binaries or compiled AVX/CUDA builds.
- A `llama.cpp` source-build manager; InferenceDeck consumes runtimes rather than owning their build toolchains.

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

## Development

```bash
python -m unittest discover -s tests -v
```

CI exercises Python 3.10 and 3.12 on Linux, Windows, and macOS.

## Provenance

InferenceDeck is a clean continuation of the portable core developed in the earlier local **Llama Control Center** project. Historical UI and environment-specific integrations remain donor material and are not part of this initial import.
