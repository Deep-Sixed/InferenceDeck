"""Run helper processes without flashing console windows on Windows.

PowerShell, tasklist, nvidia-smi and llama-server are console programs: started
from a process without a console (the tray, pythonw, a service), Windows opens a
window for each one. Every short-lived helper call goes through ``run`` so it
starts hidden.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

# Passed to every child process; empty off Windows.
NO_WINDOW: dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` that never opens a console window on Windows."""
    return subprocess.run(args, **{**NO_WINDOW, **kwargs})
