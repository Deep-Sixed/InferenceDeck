"""CPU instruction-set detection for picking a compatible llama.cpp build.

Only the x86 features llama.cpp builds are compiled against matter here:
SSE4.2, AVX, AVX2, FMA, F16C and AVX-512F. On non-x86 hosts (Apple Silicon,
ARM Linux) x86 requirements do not apply and ``x86`` is False.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import is_windows

TRACKED_FEATURES = ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f")
X86_MACHINES = {"x86_64", "amd64", "i386", "i686", "x86"}

# Windows IsProcessorFeaturePresent ids.
_PF_SSE4_2 = 38
_PF_AVX = 39
_PF_AVX2 = 40
_PF_AVX512F = 41


@dataclass(frozen=True)
class CpuFeatures:
    x86: bool
    features: frozenset[str] = field(default_factory=frozenset)
    source: str = "unknown"
    # Features implied rather than read directly (Windows can't query FMA/F16C).
    inferred: frozenset[str] = field(default_factory=frozenset)

    def has(self, feature: str) -> bool:
        return feature in self.features

    def to_dict(self) -> dict[str, Any]:
        return {
            "x86": self.x86,
            "features": sorted(self.features),
            "avx2": self.has("avx2"),
            "source": self.source,
            "inferred": sorted(self.inferred),
        }


def _normalize(flags: set[str]) -> frozenset[str]:
    aliases = {"sse4.2": "sse4_2", "avx1.0": "avx", "avx512": "avx512f"}
    out = {aliases.get(flag.lower(), flag.lower()) for flag in flags}
    return frozenset(out & set(TRACKED_FEATURES))


def parse_cpuinfo_flags(text: str) -> frozenset[str]:
    """Features from the first ``flags`` line of /proc/cpuinfo."""

    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "flags":
            return _normalize(set(value.split()))
    return frozenset()


def _linux() -> CpuFeatures:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return CpuFeatures(x86=True, source="unavailable")
    return CpuFeatures(x86=True, features=parse_cpuinfo_flags(text), source="/proc/cpuinfo")


def _windows() -> CpuFeatures:
    import ctypes

    present = ctypes.windll.kernel32.IsProcessorFeaturePresent  # type: ignore[attr-defined]
    found = {
        name
        for name, pf in (("sse4_2", _PF_SSE4_2), ("avx", _PF_AVX), ("avx2", _PF_AVX2), ("avx512f", _PF_AVX512F))
        if present(pf)
    }
    inferred: set[str] = set()
    if "avx2" in found:
        # Windows exposes no FMA/F16C query; every AVX2 CPU (Intel Haswell+,
        # AMD Excavator+) has both.
        inferred = {"fma", "f16c"}
    return CpuFeatures(
        x86=True,
        features=frozenset(found | inferred),
        source="IsProcessorFeaturePresent",
        inferred=frozenset(inferred),
    )


def _macos() -> CpuFeatures:
    sysctl = shutil.which("sysctl")
    if not sysctl:
        return CpuFeatures(x86=True, source="unavailable")
    flags: set[str] = set()
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        try:
            result = subprocess.run([sysctl, "-n", key], capture_output=True, text=True, timeout=2, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        flags.update(result.stdout.split())
    return CpuFeatures(x86=True, features=_normalize(flags), source="sysctl")


@lru_cache(maxsize=1)
def detect_cpu_features() -> CpuFeatures:
    if platform.machine().lower() not in X86_MACHINES:
        return CpuFeatures(x86=False, source=f"non-x86 ({platform.machine()})")
    try:
        if is_windows():
            return _windows()
        if platform.system() == "Darwin":
            return _macos()
        return _linux()
    except Exception:  # detection must never take the app down
        return CpuFeatures(x86=True, source="unavailable")
