"""Which command-line flags a llama-server binary accepts.

llama.cpp renames and removes flags between releases (``--mmap``/``--no-mmap``/
``--mlock`` became ``--load-mode``; ``--draft-max``/``--draft-min`` became
``--spec-draft-n-max``/``--spec-draft-n-min``), and an unknown flag makes
llama-server exit before loading anything. The arg builder asks this module
which spelling the installed build understands.

The answer comes from ``<binary> --help``, read once per binary and cached by
path + size + mtime (in memory and on disk), so a rebuilt binary is re-probed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .proc import run as run_hidden

_CACHE_FILENAME = "llama_server_flags.json"
_CACHE_VERSION = 1
_HELP_TIMEOUT_SECONDS = 15.0
# Long flags as they appear in help text: "-lm, --load-mode MODE", "--no-mmap".
_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")
_memory: dict[str, tuple[tuple[int, int], frozenset[str] | None]] = {}


def _signature(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_size, int(stat.st_mtime))


def _cache_file() -> Path | None:
    try:
        from .paths import cache_dir

        return cache_dir() / _CACHE_FILENAME
    except Exception:
        return None


def _load_disk() -> dict:
    path = _cache_file()
    if not path or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if data.get("_version") == _CACHE_VERSION else {}


def _store_disk(key: str, sig: tuple[int, int], flags: frozenset[str]) -> None:
    path = _cache_file()
    if not path:
        return
    try:
        data = _load_disk()
        data["_version"] = _CACHE_VERSION
        data[key] = {"size": sig[0], "mtime": sig[1], "flags": sorted(flags)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def parse_help_flags(text: str) -> frozenset[str]:
    return frozenset(_FLAG_RE.findall(text or ""))


def _probe(binary: str) -> frozenset[str] | None:
    try:
        result = run_hidden(
            [binary, "--help"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_HELP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    flags = parse_help_flags(f"{result.stdout or ''}\n{result.stderr or ''}")
    # A real llama-server help lists hundreds of flags; far fewer means the
    # binary crashed or isn't llama-server, so the answer is "unknown".
    return flags if "--ctx-size" in flags and "--port" in flags else None


def supported_flags(binary: str | None) -> frozenset[str] | None:
    """The long flags ``binary`` accepts, or ``None`` when that can't be told."""
    if not binary:
        return None
    key = str(Path(binary).expanduser())
    sig = _signature(key)
    if sig is None:
        return None
    cached = _memory.get(key)
    if cached and cached[0] == sig:
        return cached[1]
    disk = _load_disk().get(key)
    if disk and disk.get("size") == sig[0] and disk.get("mtime") == sig[1]:
        flags = frozenset(disk.get("flags") or [])
        _memory[key] = (sig, flags)
        return flags
    flags = _probe(key)
    # A failed probe is kept in memory only, so a later process tries again.
    _memory[key] = (sig, flags)
    if flags is not None:
        _store_disk(key, sig, flags)
    return flags
