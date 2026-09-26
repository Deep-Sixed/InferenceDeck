from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .paths import is_windows


def _run(args: list[str], timeout: float = 2.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _detect_ram_speed() -> dict[str, Any] | None:
    """Detect RAM speed, type, and bandwidth where available."""
    if is_windows():
        return _windows_ram_speed()
    elif platform.system() == "Darwin":
        return _mac_ram_speed()
    else:
        return _linux_ram_speed()


def _windows_ram_speed() -> dict[str, Any] | None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return None
    command = (
        "$mem = @(Get-CimInstance Win32_PhysicalMemory); "
        "if ($mem.Count -eq 0) { exit 0 }; "
        "[pscustomobject]@{"
        "Speed = ($mem | Select-Object -First 1).Speed; "
        "Type = ($mem | Select-Object -First 1).MemoryType; "
        "Manufacturer = ($mem | Select-Object -First 1).Manufacturer; "
        "ConfiguredSpeed = ($mem | Select-Object -First 1).ConfiguredClockSpeed; "
        "} | ConvertTo-Json -Compress"
    )
    result = _run([powershell, "-NoProfile", "-Command", command], timeout=3.0)
    if not result or result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    speed = _int_or_none(payload.get("Speed")) or _int_or_none(payload.get("ConfiguredSpeed"))
    mem_type = payload.get("Type")
    ram_type = _windows_memory_type(mem_type)
    return {
        "speed_mts": speed,
        "type": ram_type,
        "bandwidth_gbps": _calculate_ram_bandwidth(speed, ram_type),
    }


def _windows_memory_type(mem_type_code: Any) -> str | None:
    if mem_type_code is None:
        return None
    try:
        code = int(mem_type_code)
        type_map = {
            20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 30: "DDR5",
        }
        return type_map.get(code, f"RAM-{code}")
    except (ValueError, TypeError):
        return None


def _mac_ram_speed() -> dict[str, Any] | None:
    profiler = shutil.which("system_profiler")
    if not profiler:
        return None
    result = _run([profiler, "SPMemoryDataType", "-json"], timeout=4.0)
    if not result or result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    memory_modules = payload.get("SPMemoryDataType") or []
    if not memory_modules:
        return None
    speed_str = memory_modules[0].get("SPMemorySpeedKey", "") if isinstance(memory_modules[0], dict) else ""
    speed = None
    ram_type = None
    if isinstance(speed_str, str):
        match = re.search(r"(\d+)", speed_str)
        if match:
            speed = int(match.group(1))
        if "DDR4" in speed_str:
            ram_type = "DDR4"
        elif "DDR5" in speed_str:
            ram_type = "DDR5"
        elif "LPDDR" in speed_str:
            ram_type = "LPDDR"
    return {
        "speed_mts": speed,
        "type": ram_type,
        "bandwidth_gbps": _calculate_ram_bandwidth(speed, ram_type),
    }


def _linux_ram_speed() -> dict[str, Any] | None:
    dmidecode = shutil.which("dmidecode")
    if not dmidecode:
        return None
    result = _run([dmidecode, "-t", "memory"], timeout=3.0)
    if not result or result.returncode != 0:
        return None
    speed = None
    ram_type = None
    for line in result.stdout.splitlines():
        line_lower = line.lower().strip()
        if "speed" in line_lower and "unknown" not in line_lower:
            match = re.search(r"(\d+)\s*MT/s", line)
            if match:
                speed = int(match.group(1))
        if "type" in line_lower and "unknown" not in line_lower and "error" not in line_lower:
            for t in ["DDR", "DDR2", "DDR3", "DDR4", "DDR5"]:
                if t.lower() in line_lower:
                    ram_type = t
                    break
    if speed:
        return {
            "speed_mts": speed,
            "type": ram_type,
            "bandwidth_gbps": _calculate_ram_bandwidth(speed, ram_type),
        }
    return None


def _calculate_ram_bandwidth(speed_mts: int | None, ram_type: str | None) -> float | None:
    if not speed_mts or speed_mts <= 0:
        return None
    if ram_type in ("DDR", "DDR2", "DDR3"):
        return round(speed_mts * 8 / 1000, 1)
    if ram_type == "DDR4":
        return round(speed_mts * 8 / 1000, 1)
    if ram_type == "DDR5":
        return round(speed_mts * 16 / 1000, 1)
    if ram_type and "LPDDR" in ram_type:
        return round(speed_mts * 16 / 1000, 1)
    return round(speed_mts * 8 / 1000, 1)


def _windows_memory_info() -> dict[str, int | None]:
    if not is_windows():
        return {"total_bytes": None, "available_bytes": None}

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {
            "total_bytes": int(status.ullTotalPhys),
            "available_bytes": int(status.ullAvailPhys),
        }
    return {"total_bytes": None, "available_bytes": None}


def _posix_memory_info() -> dict[str, int | None]:
    if is_windows():
        return {"total_bytes": None, "available_bytes": None}
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(pages) * int(page_size)
        available = None
        if hasattr(os, "sysconf_names") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            available = int(os.sysconf("SC_AVPHYS_PAGES")) * int(page_size)
        return {"total_bytes": total, "available_bytes": available}
    except (AttributeError, OSError, ValueError):
        return {"total_bytes": None, "available_bytes": None}


def detect_memory() -> dict[str, Any]:
    memory = _windows_memory_info() if is_windows() else _posix_memory_info()
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        memory["unified"] = True
    else:
        memory["unified"] = False
    
    ram_speed = _detect_ram_speed()
    if ram_speed:
        memory["ram_speed_mts"] = ram_speed.get("speed_mts")
        memory["ram_bandwidth_gbps"] = ram_speed.get("bandwidth_gbps")
        memory["ram_type"] = ram_speed.get("type")
    else:
        memory["ram_type"] = None
        memory["ram_bandwidth_gbps"] = None
    
    return memory


def _windows_cpu_info() -> dict[str, Any]:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return {}
    command = (
        "$cpus = @(Get-CimInstance Win32_Processor); "
        "[pscustomobject]@{"
        "Name = ($cpus | Select-Object -First 1).Name; "
        "Physical = ($cpus | Measure-Object -Property NumberOfCores -Sum).Sum; "
        "Logical = ($cpus | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum"
        "} | ConvertTo-Json -Compress"
    )
    result = _run([powershell, "-NoProfile", "-Command", command], timeout=3.0)
    if not result or result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return {
        "name": payload.get("Name"),
        "physical_cores": _int_or_none(payload.get("Physical")),
        "logical_cores": _int_or_none(payload.get("Logical")),
    }


def _parse_cpuinfo(text: str) -> dict[str, Any]:
    name: str | None = None
    core_ids: set[tuple[str, str]] = set()
    phys_id = core_id = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not name and key in ("model name", "Hardware", "cpu model"):
            name = value
        elif key == "physical id":
            phys_id = value
        elif key == "core id":
            core_id = value
        elif line.strip() == "":  # blank line separates processor blocks
            if phys_id is not None and core_id is not None:
                core_ids.add((phys_id, core_id))
            phys_id = core_id = None
    if phys_id is not None and core_id is not None:
        core_ids.add((phys_id, core_id))
    return {"name": name, "physical_cores": len(core_ids) or None}


def _linux_cpu_info() -> dict[str, Any]:
    """Read CPU model and physical core count from /proc/cpuinfo."""
    try:
        return _parse_cpuinfo(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


def detect_cpu() -> dict[str, Any]:
    cpu = _windows_cpu_info() if is_windows() else {}
    if platform.system() == "Darwin":
        sysctl = shutil.which("sysctl")
        if sysctl:
            result = _run([sysctl, "-n", "machdep.cpu.brand_string"], timeout=1.5)
            if result and result.returncode == 0 and result.stdout.strip():
                cpu["name"] = result.stdout.strip()
    elif platform.system() == "Linux":
        cpu = _linux_cpu_info()
    logical = cpu.get("logical_cores") or os.cpu_count()
    name = cpu.get("name") or platform.processor() or platform.machine() or "Unknown CPU"
    return {
        "name": str(name).strip(),
        "physical_cores": cpu.get("physical_cores"),
        "logical_cores": logical,
        "architecture": platform.machine(),
    }


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not binary:
        return []
    result = _run(
        [
            binary,
            "--query-gpu=index,name,memory.total,memory.free,driver_version,clocks.current.memory,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=2.5,
    )
    if not result or result.returncode != 0:
        return []

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 5:
            continue
        total_mib = _int_or_none(parts[2])
        free_mib = _int_or_none(parts[3])
        driver_version = parts[4]
        # clocks.current.memory and clocks.max.memory may be "N/A" on some GPUs
        current_mem_clock = _int_or_none(parts[5]) if len(parts) > 5 else None
        max_mem_clock = _int_or_none(parts[6]) if len(parts) > 6 else None
        # Use max memory clock as the data rate (in MHz = MT/s for DDR)
        data_rate = max_mem_clock if max_mem_clock and max_mem_clock > 0 else current_mem_clock
        # Look up bus width and compute bandwidth from GPU name
        gpu_name = parts[1] if len(parts) > 1 else ""
        bus_width = _nvidia_bus_width_from_name(gpu_name)
        vram_bandwidth_gbps = None
        if bus_width and data_rate and data_rate > 0:
            # bus_width is in bits, data_rate in MT/s; /8 converts to bytes -> GB/s
            vram_bandwidth_gbps = round(bus_width * data_rate / 8 / 1000, 1)
        gpus.append(
            {
                "index": _int_or_none(parts[0]),
                "name": gpu_name,
                "vram_total_bytes": total_mib * 1024 * 1024 if total_mib is not None else None,
                "vram_free_bytes": free_mib * 1024 * 1024 if free_mib is not None else None,
                "vram_bus_width_bits": bus_width,
                "vram_data_rate_mts": data_rate,
                "vram_bandwidth_gbps": vram_bandwidth_gbps,
                "driver_version": driver_version,
                "backend": "nvidia-smi",
                "vendor": "NVIDIA",
                "integrated": False,
                "acceleration_backend": "cuda",
                "acceleration_options": ["cuda", "vulkan", "cpu"],
            }
        )
    return gpus


def _nvidia_bus_width_from_name(name: str) -> int | None:
    """Guess NVIDIA bus width from GPU product name."""
    if not name:
        return None
    lowered = name.lower()
    # RTX 50-series
    if any(m in lowered for m in ["rtx 5090"]):
        return 384
    if any(m in lowered for m in ["rtx 5080"]):
        return 256
    if any(m in lowered for m in ["rtx 5070 ti", "rtx 5070"]):
        return 192
    if any(m in lowered for m in ["rtx 5060"]):
        return 128
    # RTX 40-series
    if any(m in lowered for m in ["rtx 4090"]):
        return 384
    if any(m in lowered for m in ["rtx 4080 super", "rtx 4080"]):
        return 256
    if any(m in lowered for m in ["rtx 4070 ti super", "rtx 4070 ti", "rtx 4070 super", "rtx 4070"]):
        return 192
    if any(m in lowered for m in ["rtx 4060 ti 16gb", "rtx 4060 ti 8gb", "rtx 4060 ti", "rtx 4060"]):
        return 128
    # RTX 30-series
    if any(m in lowered for m in ["rtx 3090 ti", "rtx 3090"]):
        return 384
    if any(m in lowered for m in ["rtx 3080 ti", "rtx 3080 16gb", "rtx 3080"]):
        return 384
    if any(m in lowered for m in ["rtx 3070 ti", "rtx 3070"]):
        return 192
    if any(m in lowered for m in ["rtx 3060 12gb", "rtx 3060 ti", "rtx 3060"]):
        return 192
    # RTX 20-series
    if any(m in lowered for m in ["rtx 2080 ti", "rtx 2080 super", "rtx 2080"]):
        return 256
    if any(m in lowered for m in ["rtx 2070 super", "rtx 2070"]):
        return 192
    if any(m in lowered for m in ["rtx 2060"]):
        return 192
    # GTX series
    if any(m in lowered for m in ["gtx 1660 ti", "gtx 1660 super", "gtx 1660"]):
        return 192
    if any(m in lowered for m in ["gtx 1080 ti", "gtx 1080"]):
        return 256
    if any(m in lowered for m in ["gtx 1070 ti", "gtx 1070"]):
        return 192
    if any(m in lowered for m in ["gtx 1650", "gtx 1060 6gb", "gtx 1060 3gb", "gtx 1050 ti", "gtx 1050"]):
        return 128
    # Tesla / Quadro / A-series / L-series
    if any(m in lowered for m in ["a100", "a100 pcie", "a100 sfp+", "a800"]):
        return 5120 if "hbm" in lowered or "hbm2" in lowered else 384
    if any(m in lowered for m in ["a10", "a40"]):
        return 384 if "a40" in lowered else 256
    if any(m in lowered for m in ["l40", "l40s"]):
        return 384
    if any(m in lowered for m in ["l4"]):
        return 96
    if any(m in lowered for m in ["quadro", "tesla"]):
        return 256
    return None


def _guess_vram_specs(name: str, vendor: str | None, revision: str, device_id: str, pnp_id: str) -> dict[str, Any]:
    """Guess VRAM bus width and data rate from GPU name, revision, and device ID."""
    result: dict[str, Any] = {"bus_width_bits": None, "data_rate_mts": None, "bandwidth_gbps": None}
    if not revision and not device_id:
        return result
    rev_int = None
    if revision:
        try:
            rev_int = int(revision, 16)
        except (ValueError, TypeError):
            pass
    bus_width = None
    data_rate = None
    if vendor == "NVIDIA":
        bus_width = _nvidia_bus_width(device_id, rev_int)
        data_rate = _nvidia_data_rate(rev_int)
    elif vendor == "AMD":
        bus_width = _amd_bus_width(device_id, rev_int)
        data_rate = _amd_data_rate(rev_int)
    elif vendor == "Intel":
        bus_width = 64
        data_rate = None
    if bus_width and data_rate:
        result["bus_width_bits"] = bus_width
        result["data_rate_mts"] = data_rate
        # bus_width bits * data_rate MT/s, /8 for bits->bytes -> GB/s
        result["bandwidth_gbps"] = round(bus_width * data_rate / 8 / 1000, 1)
    elif bus_width:
        result["bus_width_bits"] = bus_width
    elif data_rate:
        result["data_rate_mts"] = data_rate
    return result


def _nvidia_bus_width(device_id: str, rev: int | None) -> int | None:
    if not device_id:
        return None
    dev = int(device_id, 16) if device_id else 0
    width_map = {
        # RTX 50-series (Blackwell)
        0x2784: 384, 0x2785: 384, 0x2B85: 384, 0x2789: 384, 0x278A: 384, 0x278B: 384,
        0x278C: 384, 0x278D: 256, 0x278E: 256, 0x2790: 256, 0x2791: 256,
        0x2792: 256, 0x2793: 256, 0x2794: 256, 0x2795: 256, 0x2796: 256,
        0x2797: 128, 0x2798: 128, 0x2799: 128, 0x279A: 128,
        # RTX 40-series (Ada Lovelace)
        0x2204: 256, 0x2206: 256, 0x2208: 256, 0x2209: 128, 0x220E: 256,
        0x2210: 256, 0x2211: 256, 0x2212: 128, 0x2213: 128, 0x2214: 128,
        0x2215: 128, 0x2216: 128, 0x2304: 128, 0x2306: 128, 0x2308: 128,
        0x2309: 128, 0x230A: 128, 0x230B: 128,
        # RTX 20-series (Turing)
        0x1E04: 256, 0x1E07: 256, 0x1E02: 128, 0x1E0D: 256, 0x1E94: 256,
        0x1D04: 256, 0x1D83: 128, 0x1D84: 128, 0x1D85: 128, 0x1D87: 128,
        0x1D88: 128, 0x1D89: 128, 0x1D8A: 128, 0x1D8B: 128, 0x1D8C: 128,
        0x1D8D: 128, 0x1D8F: 128, 0x1DB0: 192, 0x1DB1: 256, 0x1DB2: 192,
        # GTX 16-series (Turing)
        0x1F10: 128, 0x1F11: 128, 0x1F12: 128, 0x1F14: 128,
        0x1F91: 128, 0x1F92: 128, 0x1F93: 128, 0x1F95: 128,
        # GTX 10-series (Pascal)
        0x1B80: 256, 0x1B81: 256, 0x1B82: 256, 0x1B83: 256,
        0x1B84: 256, 0x1B86: 256, 0x1B87: 128,
        0x1C03: 256, 0x1C02: 256, 0x1C07: 256, 0x1C0D: 256, 0x1C0F: 256,
        0x1A10: 256, 0x1A11: 256, 0x1A12: 256, 0x1A14: 256,
        0x1A18: 256, 0x1A19: 256, 0x1A1C: 128, 0x1A1D: 128, 0x1A1F: 128,
        # Tesla / Quadro (various)
        0x1DBF: 256, 0x1DBE: 256, 0x1DBD: 256, 0x1DBC: 256,
        0x1D7C: 48, 0x1D7D: 48, 0x1D7E: 96,
    }
    return width_map.get(dev)


def _nvidia_data_rate(rev: int | None) -> int | None:
    if rev is None:
        return None
    rate_map = {
        # Common revisions
        0xA1: 3500, 0xA0: 3000, 0xA2: 4000,
        0x01: 2500, 0x02: 3000, 0x03: 3500,
        0x04: 4000, 0x05: 4200,
        # RTX 50-series revisions
        0xC0: 18000, 0xC1: 18000, 0xC2: 18000,
        0xD0: 16000, 0xD1: 16000,
    }
    return rate_map.get(rev)


def _amd_bus_width(device_id: str, rev: int | None) -> int | None:
    if not device_id:
        return None
    dev = int(device_id, 16) if device_id else 0
    width_map = {
        # RX 9000 series (RDNA 4)
        0x1AAA: 128, 0x1AAB: 128, 0x1AAC: 128, 0x1AAD: 128,
        0x1B00: 128, 0x1B01: 128, 0x1B02: 128, 0x1B03: 128,
        # RX 8000 series (RDNA 4)
        0x173F: 128, 0x173E: 128, 0x173D: 128, 0x173C: 128,
        # RX 7000 series (RDNA 3)
        0x743E: 256, 0x743F: 256, 0x743C: 128,
        0x741F: 256, 0x741E: 256, 0x741D: 256, 0x741C: 128,
        0x7408: 128, 0x7409: 128, 0x740A: 128, 0x740B: 128,
        0x7420: 128, 0x7421: 128, 0x7422: 128, 0x7423: 128,
        0x7424: 128, 0x7425: 128, 0x7426: 128, 0x7427: 128,
        # RX 6000 series (RDNA 2)
        0x73DF: 256, 0x73DE: 256, 0x73DB: 128,
        0x73A0: 256, 0x73A1: 256, 0x73A2: 256, 0x73A3: 256,
        0x73BF: 256, 0x73BE: 256, 0x73BD: 128,
        # RX 5000 series (RDNA)
        0x164C: 128, 0x164D: 128, 0x164E: 128,
        0x15DD: 256, 0x15DE: 256, 0x15DF: 256,
        0x15D8: 128, 0x15D9: 128, 0x15DA: 128, 0x15DB: 128,
    }
    return width_map.get(dev)


def _amd_data_rate(rev: int | None) -> int | None:
    if rev is None:
        return None
    rate_map = {
        0xA1: 3500, 0xA0: 3000, 0x01: 2500, 0x02: 3000,
        0xB0: 18000, 0xB1: 18000, 0xB2: 16000,
        0xC0: 20000, 0xC1: 20000,
    }
    return rate_map.get(rev)


def _linux_lspci_bus_width(pci_bus: str, name: str, vendor: str | None) -> int | None:
    """Guess bus width from GPU name for Linux lspci fallback."""
    if not vendor:
        return None
    lowered = name.lower()
    if vendor == "NVIDIA":
        if any(m in lowered for m in ["rtx 4090", "rtx 4080 super", "rtx 4080", "rtx 3090 ti", "rtx 3090", "a100", "h100"]):
            return 384
        if any(m in lowered for m in ["rtx 4070", "rtx 3080 ti", "rtx 3080", "rtx 2080 ti", "rtx 2080", "titan rtx"]):
            return 256
        if any(m in lowered for m in ["rtx 4060", "rtx 3070", "rtx 3060", "rtx 2070", "gtx 1660", "gtx 1080", "gtx 1070"]):
            return 192
        if any(m in lowered for m in ["gt 1030", "gt 1050", "gtx 1650", "gtx 750"]):
            return 128
        if any(m in lowered for m in ["quadro", "tesla"]):
            return 256
    elif vendor == "AMD":
        if any(m in lowered for m in ["rx 7900 xtx", "rx 7900 xt", "rx 6900 xt"]):
            return 256
        if any(m in lowered for m in ["rx 7800 xt", "rx 7700 xt", "rx 6800 xt", "rx 6800"]):
            return 256
        if any(m in lowered for m in ["rx 6700 xt", "rx 6600 xt", "rx 5700 xt"]):
            return 256
        if any(m in lowered for m in ["rx 6600", "rx 5600", "rx 5500"]):
            return 128
    elif vendor == "Intel":
        if any(m in lowered for m in ["arc a770", "arc a750"]):
            return 256
        if any(m in lowered for m in ["arc a380"]):
            return 128
    return None


def _linux_lspci_data_rate(vendor: str | None, bus_width: int | None) -> int | None:
    """Guess data rate from vendor for Linux lspci fallback."""
    if not vendor:
        return None
    lowered = vendor.lower()
    if "nvidia" in lowered:
        if bus_width == 384:
            return 17500
        if bus_width == 256:
            return 16000
        if bus_width == 192:
            return 14000
        if bus_width == 128:
            return 12000
    elif "amd" in lowered:
        if bus_width == 256:
            return 18000
        if bus_width == 128:
            return 16000
    elif "intel" in lowered:
        return 16000
    return None


def _gpu_vendor(name: str) -> str | None:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "rtx" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "radeon" in lowered or "firepro" in lowered:
        return "AMD"
    if "intel" in lowered or "arc" in lowered or "iris" in lowered or "uhd" in lowered:
        return "Intel"
    if "apple" in lowered or "m1" in lowered or "m2" in lowered or "m3" in lowered or "m4" in lowered:
        return "Apple"
    return None


def _gpu_integrated(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ["uhd", "iris", "vega", "integrated", "apple", "apu"])


def _acceleration_options(vendor: str | None, system: str | None = None) -> list[str]:
    system = system or platform.system()
    if system == "Darwin":
        return ["metal", "cpu"]
    if vendor == "NVIDIA":
        return ["cuda", "vulkan", "cpu"]
    if vendor == "AMD":
        return ["vulkan", "hip", "cpu"]
    if vendor == "Intel":
        return ["vulkan", "sycl", "cpu"]
    return ["cpu"]


def _preferred_backend(vendor: str | None, system: str | None = None) -> str:
    options = _acceleration_options(vendor, system)
    return options[0] if options else "cpu"


def _is_virtual_display(name: str) -> bool:
    lowered = name.lower()
    virtual_markers = [
        "virtual",
        "remote",
        "parsec",
        "mirage",
        "basic display",
        "meta virtual monitor",
        "spacedesk",
    ]
    return any(marker in lowered for marker in virtual_markers)


def _windows_display_gpus() -> list[dict[str, Any]]:
    if not is_windows():
        return []
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return []
    command = (
        "$gpus = Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -match 'NVIDIA|AMD|Radeon|Intel|Arc|Iris|UHD' } | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID; "
        "foreach ($g in $gpus) { "
        "  $pnp = $g.PNPDeviceID; "
        "  $venMatch = [regex]::Match($pnp, 'VEN_([0-9A-Fa-f]{4})'); "
        "  $devMatch = [regex]::Match($pnp, 'DEV_([0-9A-Fa-f]{4})'); "
        "  $revMatch = [regex]::Match($pnp, 'REV_([0-9A-Fa-f]+)'); "
        "  $vendorId = if ($venMatch.Success) { $venMatch.Groups[1].Value } else { '' }; "
        "  $deviceId = if ($devMatch.Success) { $devMatch.Groups[1].Value } else { '' }; "
        "  $rev = if ($revMatch.Success) { $revMatch.Groups[1].Value } else { '' }; "
        "  [pscustomobject]@{"
        "    Name = $g.Name; "
        "    AdapterRAM = $g.AdapterRAM; "
        "    DriverVersion = $g.DriverVersion; "
        "    PNPDeviceID = $g.PNPDeviceID; "
        "    VendorId = $vendorId; "
        "    DeviceId = $deviceId; "
        "    Revision = $rev; "
        "  } "
        "} | ConvertTo-Json -Compress"
    )
    result = _run([powershell, "-NoProfile", "-Command", command], timeout=3.0)
    if not result or result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    gpus: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "Display adapter").strip()
        if not name or _is_virtual_display(name):
            continue
        vendor = _gpu_vendor(name)
        adapter_ram = _int_or_none(item.get("AdapterRAM"))
        revision = str(item.get("Revision") or "")
        device_id = str(item.get("DeviceId") or "")
        pnp_id = str(item.get("PNPDeviceID") or "")
        vram_info = _guess_vram_specs(name, vendor, revision, device_id, pnp_id)
        gpus.append(
            {
                "index": idx,
                "name": name,
                "vram_total_bytes": adapter_ram if adapter_ram and adapter_ram > 0 else None,
                "vram_free_bytes": None,
                "vram_data_rate_mts": vram_info.get("data_rate_mts"),
                "vram_bus_width_bits": vram_info.get("bus_width_bits"),
                "vram_bandwidth_gbps": vram_info.get("bandwidth_gbps"),
                "driver_version": item.get("DriverVersion"),
                "backend": "windows-cim",
                "vendor": vendor,
                "integrated": _gpu_integrated(name),
                "acceleration_backend": _preferred_backend(vendor),
                "acceleration_options": _acceleration_options(vendor),
            }
        )
    return gpus


def _mac_display_gpus() -> list[dict[str, Any]]:
    if platform.system() != "Darwin":
        return []
    profiler = shutil.which("system_profiler")
    if not profiler:
        return []
    result = _run([profiler, "SPDisplaysDataType", "-json"], timeout=4.0)
    if not result or result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    items = payload.get("SPDisplaysDataType") or []
    gpus: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = item.get("sppci_model") or item.get("_name") or item.get("spdisplays_vendor") or "Apple GPU"
        vendor = _gpu_vendor(str(name)) or "Apple"
        vram_text = " ".join(
            str(item.get(key, "")) for key in ["spdisplays_vram", "spdisplays_vram_shared", "spdisplays_vram_dynamic"]
        )
        vram_bytes = None
        match = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", vram_text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            vram_bytes = int(value * (1024**3 if unit == "gb" else 1024**2))
        vram_info = _guess_vram_specs(str(name), vendor, "", "", "")
        gpus.append(
            {
                "index": idx,
                "name": str(name),
                "vram_total_bytes": vram_bytes,
                "vram_free_bytes": None,
                "vram_data_rate_mts": vram_info.get("data_rate_mts"),
                "vram_bus_width_bits": vram_info.get("bus_width_bits"),
                "vram_bandwidth_gbps": vram_info.get("bandwidth_gbps"),
                "driver_version": None,
                "backend": "system-profiler",
                "vendor": vendor,
                "integrated": True,
                "unified_memory": True,
                "acceleration_backend": "metal",
                "acceleration_options": ["metal", "cpu"],
            }
        )
    return gpus


def _linux_lspci_gpus() -> list[dict[str, Any]]:
    if platform.system() == "Darwin" or is_windows():
        return []
    lspci = shutil.which("lspci")
    if not lspci:
        return []
    result = _run([lspci], timeout=2.0)
    if not result or result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not re.search(r"(?i)(vga|3d controller|display controller)", line):
            continue
        name = re.sub(r"^[0-9a-fA-F:.]+\s+", "", line).strip()
        # Drop the "VGA compatible controller: " device-class prefix and "(rev NN)" suffix.
        name = re.sub(r"^[^:]+:\s*", "", name)
        name = re.sub(r"\s*\(rev [0-9a-fA-F]+\)\s*$", "", name).strip()
        if _is_virtual_display(name):
            continue
        vendor = _gpu_vendor(name)
        pci_addr = re.match(r"^([0-9a-fA-F:.]+)", line)
        pci_bus = pci_addr.group(1).split(":")[0] if pci_addr else ""
        vram_total = None
        vram_info = _guess_vram_specs(name, vendor, "", "", "")
        vram_info["bus_width_bits"] = _linux_lspci_bus_width(pci_bus, name, vendor)
        vram_info["data_rate_mts"] = _linux_lspci_data_rate(vendor, vram_info.get("bus_width_bits"))
        if vram_info["bus_width_bits"] and vram_info["data_rate_mts"]:
            vram_info["bandwidth_gbps"] = round(vram_info["bus_width_bits"] * vram_info["data_rate_mts"] / 8 / 1000, 1)
        gpus.append(
            {
                "index": len(gpus),
                "name": name,
                "vram_total_bytes": vram_total,
                "vram_free_bytes": None,
                "vram_data_rate_mts": vram_info.get("data_rate_mts"),
                "vram_bus_width_bits": vram_info.get("bus_width_bits"),
                "vram_bandwidth_gbps": vram_info.get("bandwidth_gbps"),
                "driver_version": None,
                "backend": "lspci",
                "vendor": vendor,
                "integrated": _gpu_integrated(name),
                "acceleration_backend": _preferred_backend(vendor, platform.system()),
                "acceleration_options": _acceleration_options(vendor, platform.system()),
            }
        )
    return gpus


def _dedupe_gpus(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for gpu in gpus:
        name = str(gpu.get("name") or "").lower()
        vendor = str(gpu.get("vendor") or "").lower()
        key = re.sub(r"[^a-z0-9]+", "", f"{vendor}:{name}")
        if not key or key in seen:
            continue
        seen.add(key)
        gpu["index"] = len(deduped)
        deduped.append(gpu)
    return deduped


def detect_gpus() -> list[dict[str, Any]]:
    gpus = _nvidia_smi_gpus()
    if is_windows():
        gpus.extend(_windows_display_gpus())
    elif platform.system() == "Darwin":
        gpus.extend(_mac_display_gpus())
    else:
        gpus.extend(_linux_lspci_gpus())
    return _dedupe_gpus(gpus)


def recommended_headroom_mib(gpus: list[dict[str, Any]]) -> int:
    primary = next((gpu for gpu in gpus if gpu.get("vram_total_bytes")), None)
    if not primary:
        return 1024
    total_mib = int(primary["vram_total_bytes"] / 1024 / 1024)
    return max(1024, min(4096, int(round(total_mib * 0.06 / 256) * 256)))


def detect_system_hardware() -> dict[str, Any]:
    gpus = detect_gpus()
    memory = detect_memory()
    primary_gpu = gpus[0] if gpus else None
    warnings: list[str] = []
    if not gpus:
        warnings.append("No GPU was detected. NVIDIA VRAM details require nvidia-smi; AMD/Intel fallback uses OS display APIs.")
    if memory.get("total_bytes") is None:
        warnings.append("System memory size could not be detected on this platform.")

    return {
        "cpu": detect_cpu(),
        "memory": memory,
        "gpus": gpus,
        "primary_gpu": primary_gpu,
        "recommended_fit_target_mib": recommended_headroom_mib(gpus),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "warnings": warnings,
    }
