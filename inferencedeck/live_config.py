"""Keep using the last good version of a config file while it is being edited.

InferenceDeck reads its configuration files on every use, so an edit takes
effect without a restart. The flip side used to be that a file caught
half-saved, or saved with a typo, took effect just as fast: a ``config.json``
that stopped being valid JSON meant every request ran on defaults, and a broken
``models.json`` made the profile list fail.

A ``LiveFile`` reads the file each time but parses and validates it only when
its bytes change, and swaps in the new version only if that succeeds. A
rejected version leaves the previous good one in use and is reported (see
``rejected_files``) until the file is fixed. A file that is deleted means
"back to defaults", as before; a file that was never valid has no previous
version, so callers fall back to their defaults and the rejection says so.
"""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Rejected(ValueError):
    """Raised by a parser: this version of the file must not replace the last good one."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LiveFile(Generic[T]):
    def __init__(self, path: Path, parse: Callable[[bytes], T], label: str) -> None:
        self.path = path
        self.label = label
        self._parse = parse
        self._lock = threading.Lock()
        self._good_bytes: bytes | None = None
        self._good: T | None = None
        self._loaded_at: str | None = None
        self._bad_bytes: bytes | None = None
        self._rejection: dict[str, Any] | None = None

    def get(self) -> T | None:
        """The current good version (a copy), or None when there is none (missing or never valid)."""
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            with self._lock:
                self._good_bytes = self._good = self._loaded_at = self._bad_bytes = self._rejection = None
            return None
        except OSError as exc:
            with self._lock:
                self._rejection = {"at": _now(), "errors": [f"cannot be read: {exc}"]}
                return copy.deepcopy(self._good)
        with self._lock:
            if data == self._good_bytes:
                # Unchanged, or reverted to the good version: nothing is rejected.
                self._bad_bytes = self._rejection = None
            elif data != self._bad_bytes:
                try:
                    value = self._parse(data)
                except Rejected as exc:
                    self._bad_bytes = data
                    self._rejection = {"at": _now(), "errors": exc.errors}
                else:
                    self._good_bytes, self._good, self._loaded_at = data, value, _now()
                    self._bad_bytes = self._rejection = None
            return copy.deepcopy(self._good)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "file": str(self.path),
                "label": self.label,
                "loaded_at": self._loaded_at,
                "rejected": copy.deepcopy(self._rejection),
                # What is in effect while the file is rejected.
                "using": "previous version" if self._good_bytes is not None else "defaults",
            }


_registry: dict[tuple[str, str], LiveFile[Any]] = {}
_registry_lock = threading.Lock()


def live_file(path: str | Path, parse: Callable[[bytes], T], label: str) -> LiveFile[T]:
    """The shared LiveFile for ``path`` read with ``parse`` (one per file and kind)."""
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    key = (str(resolved), label)
    with _registry_lock:
        entry = _registry.get(key)
        if entry is None:
            entry = _registry[key] = LiveFile(resolved, parse, label)
        return entry


def rejected_files() -> list[dict[str, Any]]:
    """Every tracked file whose latest version was rejected and not yet fixed."""
    with _registry_lock:
        entries = list(_registry.values())
    return [state for state in (entry.state() for entry in entries) if state["rejected"]]


def reset() -> None:
    """Forget every tracked file (tests)."""
    with _registry_lock:
        _registry.clear()
