"""Safe writes for the small JSON files InferenceDeck keeps.

The web UI serves requests on parallel threads and the CLI can run next to it,
so a read-modify-write of a shared file needs a lock, and a write needs a
temp file of its own before it replaces the real one.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .paths import is_windows


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a unique temp file, so concurrent writers never share one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        _replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _replace(src: str, dst: Path) -> None:
    # Windows refuses to replace a file another thread or process has open
    # (e.g. a reader mid-load); that lasts milliseconds, so retry briefly.
    for attempt in range(40):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not is_windows() or attempt == 39:
                raise
            time.sleep(0.05)


def lock_file(fh: Any) -> None:
    if is_windows():
        import msvcrt

        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.05)
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)


def unlock_file(fh: Any) -> None:
    if is_windows():
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_HELD = threading.local()


@contextmanager
def locked(lock_path: Path) -> Iterator[None]:
    """Hold ``lock_path`` against other threads and processes; re-entrant within a thread."""
    key = os.path.abspath(lock_path)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        held = getattr(_HELD, "paths", None)
        if held is None:
            held = _HELD.paths = set()
        if key in held:
            yield  # this thread already holds the file lock
            return
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as fh:
            lock_file(fh)
            held.add(key)
            try:
                yield
            finally:
                held.discard(key)
                unlock_file(fh)
