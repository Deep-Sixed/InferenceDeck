"""Follow a server's log files for live streaming, with bounded memory.

Servers write straight to their log files (see start_profile), so a slow or
stalled viewer can never back-pressure the inference process: the file on disk
stays the full record and this module only reads it. What it bounds is the
viewer side:

- history: a new viewer gets at most ``history_bytes`` of each file, starting
  at a line boundary, instead of the whole file;
- backlog: if a file grows faster than the viewer takes it, the follower jumps
  ahead and reports how many bytes it skipped instead of buffering them;
- partial lines: a line with no newline yet is held back, up to ``max_line``
  bytes, then emitted as is.

A file that shrinks or is replaced (the server was restarted, which truncates
its logs) is reported as a reset and followed from the start.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_BYTES = 64 * 1024
DEFAULT_MAX_BACKLOG_BYTES = 1024 * 1024
DEFAULT_READ_BYTES = 256 * 1024
DEFAULT_MAX_LINE = 16 * 1024


def tail_bytes(path: str | Path | None, max_bytes: int) -> str:
    """The last ``max_bytes`` of a text file, starting at a line boundary."""
    if not path:
        return ""
    try:
        with open(path, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            start = max(0, size - max_bytes)
            fh.seek(start)
            data = fh.read(max_bytes)
    except OSError:
        return ""
    if start > 0:
        newline = data.find(b"\n")
        data = data[newline + 1 :] if newline >= 0 else b""
    return data.decode("utf-8", errors="replace")


@dataclass
class _Tracked:
    name: str
    path: Path
    offset: int = 0
    identity: tuple[int, int] | None = None
    partial: bytes = b""
    # After a skip the offset lands mid-line; drop bytes up to the next newline.
    resync: bool = False


@dataclass
class LogFollower:
    """Reads new lines from one or more named log files on each ``poll()``."""

    paths: dict[str, str | Path]
    history_bytes: int = DEFAULT_HISTORY_BYTES
    max_backlog_bytes: int = DEFAULT_MAX_BACKLOG_BYTES
    read_bytes: int = DEFAULT_READ_BYTES
    max_line: int = DEFAULT_MAX_LINE
    _files: list[_Tracked] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._files = [_Tracked(name, Path(path)) for name, path in self.paths.items() if path]

    @staticmethod
    def _stat(path: Path) -> tuple[int, tuple[int, int]] | None:
        try:
            st = path.stat()
        except OSError:
            return None
        return st.st_size, (st.st_dev, st.st_ino)

    def start(self) -> list[dict[str, Any]]:
        """Recent history of each file; following continues from its end."""
        events: list[dict[str, Any]] = []
        for tracked in self._files:
            stat = self._stat(tracked.path)
            if stat is None:
                continue
            size, identity = stat
            tracked.identity = identity
            tracked.offset = size
            text = tail_bytes(tracked.path, self.history_bytes)
            lines = text.splitlines()
            if lines:
                events.append({"type": "lines", "stream": tracked.name, "lines": lines, "history": True})
        return events

    def poll(self) -> list[dict[str, Any]]:
        """New lines (and reset/skipped notices) since the last call."""
        events: list[dict[str, Any]] = []
        for tracked in self._files:
            events.extend(self._poll_one(tracked))
        return events

    def _poll_one(self, tracked: _Tracked) -> list[dict[str, Any]]:
        stat = self._stat(tracked.path)
        if stat is None:
            return []
        size, identity = stat
        events: list[dict[str, Any]] = []
        if size < tracked.offset or (tracked.identity is not None and identity != tracked.identity):
            events.append({"type": "reset", "stream": tracked.name})
            tracked.offset = 0
            tracked.partial = b""
        tracked.identity = identity
        backlog = size - tracked.offset
        if backlog > self.max_backlog_bytes:
            # Too far behind: drop the middle rather than buffer it.
            skip_to = size - self.history_bytes
            events.append({"type": "skipped", "stream": tracked.name, "bytes": skip_to - tracked.offset})
            tracked.offset = skip_to
            tracked.partial = b""
            tracked.resync = True
        if size <= tracked.offset:
            return events
        try:
            with open(tracked.path, "rb") as fh:
                fh.seek(tracked.offset)
                data = fh.read(min(size - tracked.offset, self.read_bytes))
        except OSError:
            return events
        tracked.offset += len(data)
        if tracked.resync:
            newline = data.find(b"\n")
            if newline < 0:
                return events
            data, tracked.resync = data[newline + 1 :], False
        data = tracked.partial + data
        cut = data.rfind(b"\n")
        if cut < 0:
            complete, tracked.partial = b"", data
        else:
            complete, tracked.partial = data[: cut + 1], data[cut + 1 :]
        if len(tracked.partial) > self.max_line:
            complete, tracked.partial = complete + tracked.partial, b""
        if complete:
            lines = complete.decode("utf-8", errors="replace").splitlines()
            events.append({"type": "lines", "stream": tracked.name, "lines": lines})
        return events
