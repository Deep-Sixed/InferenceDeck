"""Discover llama.cpp builds and pick the best one this CPU can run.

Policy:

1. A pinned runtime (config ``llama_runtime``), if it is compatible.
2. A standard build. Standard builds need AVX2 unless they say otherwise.
3. An AVX1 compatibility build with CUDA, when an NVIDIA GPU is present.
4. A CPU-only AVX1 compatibility build.
5. Otherwise fail with the reason each build was rejected.

A build's requirements come from, in order: an ``inferencedeck-runtime.json``
sidecar next to the binary, the build's ``CMakeCache.txt``, or the default
(standard, needs AVX2). Nothing ever launches a build whose instruction set
the CPU lacks, whatever is pinned.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cpu_features import CpuFeatures, detect_cpu_features
from .paths import executable_names

SIDECAR_NAME = "inferencedeck-runtime.json"
AUTO = "auto"

STANDARD = "standard"
CUDA_AVX1 = "cuda-avx1"
CPU_AVX1 = "cpu-avx1"

LABELS = {
    STANDARD: "Standard llama.cpp",
    CUDA_AVX1: "CUDA AVX1 compatibility build",
    CPU_AVX1: "CPU AVX1 compatibility build",
}
FEATURE_NAMES = {"sse4_2": "SSE4.2", "avx": "AVX", "avx2": "AVX2", "fma": "FMA", "f16c": "F16C", "avx512f": "AVX-512F"}
# Build-time switches in CMakeCache.txt -> CPU feature they require. Older
# llama.cpp trees used LLAMA_* names.
CMAKE_ISA_FLAGS = {
    "AVX": "avx",
    "AVX2": "avx2",
    "FMA": "fma",
    "F16C": "f16c",
    "AVX512": "avx512f",
}


@dataclass
class RuntimeCandidate:
    path: str
    variant: str
    requires: frozenset[str]
    gpu_backend: str | None
    source: str
    label: str = ""
    notes: list[str] = field(default_factory=list)
    pinned: bool = False
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "variant": self.variant,
            "label": self.label or LABELS.get(self.variant, self.variant),
            "requires": sorted(self.requires),
            "gpu_backend": self.gpu_backend,
            "source": self.source,
            "notes": list(self.notes),
        }


def _feature_list(features: frozenset[str] | set[str]) -> str:
    return ", ".join(FEATURE_NAMES.get(f, f) for f in sorted(features))


def parse_cmake_cache(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.split(":", 1)[0].strip()] = value.strip()
    return values


def _on(values: dict[str, str], *names: str) -> bool | None:
    for name in names:
        if name in values:
            return values[name].upper() in {"ON", "TRUE", "1", "YES"}
    return None


def _classify_from_cmake(values: dict[str, str]) -> tuple[frozenset[str], str | None, list[str]] | None:
    cuda = _on(values, "GGML_CUDA", "LLAMA_CUDA", "LLAMA_CUBLAS", "GGML_CUBLAS")
    gpu = "cuda" if cuda else None
    notes: list[str] = []
    if _on(values, "GGML_CPU_ALL_VARIANTS") and _on(values, "GGML_BACKEND_DL"):
        # Ships every CPU variant and picks one at load time: runs anywhere.
        notes.append("portable build: selects its CPU code path at load time")
        return frozenset(), gpu, notes
    if _on(values, "GGML_NATIVE", "LLAMA_NATIVE"):
        # Built for whatever CPU compiled it; the cache doesn't say which.
        notes.append("native build: CPU requirements unknown, assumed to need AVX2")
        return frozenset({"avx", "avx2"}), gpu, notes
    requires = set()
    known = False
    for suffix, feature in CMAKE_ISA_FLAGS.items():
        state = _on(values, f"GGML_{suffix}", f"LLAMA_{suffix}")
        if state is not None:
            known = True
            if state:
                requires.add(feature)
    if not known and gpu is None:
        return None
    return frozenset(requires), gpu, notes


def _variant_for(requires: frozenset[str], gpu: str | None) -> str:
    if "avx2" in requires or "avx" not in requires:
        return STANDARD
    return CUDA_AVX1 if gpu == "cuda" else CPU_AVX1


def classify_binary(binary: Path) -> RuntimeCandidate:
    notes: list[str] = []
    sidecar = binary.parent / SIDECAR_NAME
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            requires = frozenset(str(f).lower() for f in (data.get("cpu") or {}).get("requires", []))
            gpu = (data.get("gpu") or {}).get("backend")
            gpu = str(gpu).lower() if gpu else None
            return RuntimeCandidate(
                path=str(binary),
                variant=str(data.get("variant") or _variant_for(requires, gpu)),
                requires=requires,
                gpu_backend=gpu,
                source=SIDECAR_NAME,
                label=str(data.get("label") or ""),
            )
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            notes.append(f"ignored unreadable {SIDECAR_NAME}: {exc}")
    # bin/llama-server usually sits one level below the CMake build directory.
    for build_dir in (binary.parent, binary.parent.parent):
        cache = build_dir / "CMakeCache.txt"
        if cache.is_file():
            try:
                parsed = _classify_from_cmake(parse_cmake_cache(cache.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                parsed = None
            if parsed is not None:
                requires, gpu, cmake_notes = parsed
                return RuntimeCandidate(
                    path=str(binary),
                    variant=_variant_for(requires, gpu),
                    requires=requires,
                    gpu_backend=gpu,
                    source="CMakeCache.txt",
                    notes=notes + cmake_notes,
                )
    notes.append("no build metadata: assumed to be a standard build that needs AVX2")
    return RuntimeCandidate(
        path=str(binary),
        variant=STANDARD,
        requires=frozenset({"avx", "avx2"}),
        gpu_backend=None,
        source="default",
        notes=notes,
    )


def _binaries_under(root: Path) -> list[Path]:
    dirs = [root, root / "bin", root / "build" / "bin"]
    try:
        dirs += sorted(p / "bin" for p in root.glob("build*") if p.is_dir())
    except OSError:
        pass
    found = []
    for directory in dirs:
        for name in executable_names("llama-server"):
            candidate = directory / name
            if candidate.is_file():
                found.append(candidate)
    return found


def discover_runtimes(roots: list[Path], pinned_paths: list[str] | None = None) -> list[RuntimeCandidate]:
    seen: set[str] = set()
    candidates: list[RuntimeCandidate] = []

    def add(binary: Path, pinned: bool = False) -> None:
        try:
            key = str(binary.resolve())
        except OSError:
            key = str(binary)
        if key in seen:
            if pinned:
                for c in candidates:
                    if c.path == str(binary) or str(Path(c.path).resolve()) == key:
                        c.pinned = True
            return
        seen.add(key)
        candidate = classify_binary(binary)
        candidate.pinned = pinned
        candidates.append(candidate)

    for raw in pinned_paths or []:
        path = Path(raw).expanduser()
        if path.is_file():
            add(path, pinned=True)
    for root in roots:
        for binary in _binaries_under(root):
            add(binary)
    for name in executable_names("llama-server"):
        found = shutil.which(name)
        if found:
            add(Path(found))

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.variant] = counts.get(c.variant, 0) + 1
        c.id = c.variant if counts[c.variant] == 1 else f"{c.variant}-{counts[c.variant]}"
    return candidates


def incompatibility(candidate: RuntimeCandidate, cpu: CpuFeatures, cuda_available: bool) -> str | None:
    """Why ``candidate`` can't run on this machine, or None if it can."""

    if cpu.x86:
        if cpu.source == "unavailable" and candidate.requires:
            return "could not read this CPU's instruction sets"
        missing = candidate.requires - cpu.features
        if missing:
            return f"needs {_feature_list(missing)}, which this CPU lacks"
    if candidate.gpu_backend == "cuda" and not cuda_available:
        return "needs an NVIDIA GPU with CUDA; none detected"
    return None


def _tier(candidate: RuntimeCandidate) -> int:
    # Standard beats any compatibility build; CUDA beats CPU-only.
    if candidate.variant == STANDARD:
        return 2
    return 1 if candidate.gpu_backend else 0


def _matches(candidate: RuntimeCandidate, wanted: str) -> bool:
    return wanted in {candidate.id, candidate.variant, candidate.path}


def select_runtime(
    candidates: list[RuntimeCandidate],
    cpu: CpuFeatures,
    cuda_available: bool,
    requested: str = AUTO,
) -> dict[str, Any]:
    requested = (requested or AUTO).strip() or AUTO
    rows = []
    compatible: list[RuntimeCandidate] = []
    for c in candidates:
        why_not = incompatibility(c, cpu, cuda_available)
        rows.append({**c.to_dict(), "compatible": why_not is None, "incompatible_reason": why_not})
        if why_not is None:
            compatible.append(c)

    warnings: list[str] = []
    chosen: RuntimeCandidate | None = None
    pin = next((c for c in candidates if c.pinned), None)
    if requested != AUTO:
        pin = next((c for c in candidates if _matches(c, requested)), None)
        if pin is None:
            warnings.append(f"Pinned runtime '{requested}' was not found; selecting automatically.")
    if pin is not None:
        if pin in compatible:
            chosen = pin
        else:
            warnings.append(
                f"Pinned runtime {pin.id} can't run here ({incompatibility(pin, cpu, cuda_available)}); "
                "selecting automatically."
            )
    if chosen is None and compatible:
        # Stable sort keeps discovery order within a tier.
        chosen = sorted(compatible, key=lambda c: -_tier(c))[0]

    avx2 = "AVX2 supported" if cpu.has("avx2") else "AVX2 not supported"
    cpu_line = avx2 if cpu.x86 else "not x86: instruction-set checks don't apply"
    if chosen is None:
        if not candidates:
            reason = "No llama-server build was found."
        else:
            reason = "No compatible llama-server build: " + "; ".join(
                f"{r['id']} {r['incompatible_reason']}" for r in rows
            ) + "."
    elif chosen is pin:
        reason = f"Using pinned runtime {chosen.id}."
    elif chosen.variant == STANDARD:
        reason = f"CPU: {cpu_line}. Using the standard llama.cpp build."
    else:
        standard_rows = [r for r in rows if r["variant"] == STANDARD]
        why = standard_rows[0]["incompatible_reason"] if standard_rows else None
        prefix = f"The standard build {why}" if why else "No standard build is available"
        reason = f"CPU: {cpu_line}. {prefix}, so using the {LABELS.get(chosen.variant, chosen.variant)}."

    for row in rows:
        row["selected"] = chosen is not None and row["path"] == chosen.path
    return {
        "requested": requested,
        "policy": "pinned" if chosen is not None and chosen is pin else AUTO,
        "cpu": cpu.to_dict(),
        "cuda_available": cuda_available,
        "selected": next((r for r in rows if r["selected"]), None),
        "reason": reason,
        "candidates": rows,
        "warnings": warnings,
    }


def cuda_available() -> bool:
    from .hardware import _nvidia_smi_gpus

    try:
        return bool(_nvidia_smi_gpus())
    except Exception:
        return False


def resolve_llama_runtime(
    roots: list[Path],
    requested: str = AUTO,
    pinned_paths: list[str] | None = None,
    cpu: CpuFeatures | None = None,
    has_cuda: bool | None = None,
) -> dict[str, Any]:
    candidates = discover_runtimes(roots, pinned_paths)
    return select_runtime(
        candidates,
        cpu or detect_cpu_features(),
        cuda_available() if has_cuda is None else has_cuda,
        requested,
    )


def sibling_tool(server_path: str | None, base_name: str) -> str | None:
    """A companion binary (llama-fit-params, llama-cli) from the same build as ``server_path``."""

    if not server_path:
        return None
    directory = Path(server_path).parent
    for name in executable_names(base_name):
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return None
