from __future__ import annotations

import re
from pathlib import Path
from typing import Any


WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n\s']+\\)*[^\\/:*?\"<>|\r\n\s']*")
POSIX_HOME_PATH_RE = re.compile(r"/home/[^/\s'\"]+/(?:[^\s'\"]+)")
USER_PROFILE_RE = re.compile(r"(?i)\\Users\\([^\\/\s]+)\\")


def _looks_private_path(value: str) -> bool:
    if USER_PROFILE_RE.search(value):
        return True
    if value.startswith("/home/"):
        return True
    return False


def scan_portability_issues(project_root: Path | None) -> list[dict[str, Any]]:
    """Find obvious user-specific absolute paths in legacy config/scripts."""

    if not project_root or not project_root.is_dir():
        return []

    candidates: list[Path] = []
    candidates.extend(project_root.glob("*.ps1"))
    candidates.extend(project_root.glob("*.json"))
    scripts_dir = project_root / "scripts"
    if scripts_dir.is_dir():
        candidates.extend(scripts_dir.glob("*.ps1"))
        candidates.extend(scripts_dir.glob("*.json"))
        candidates.extend(scripts_dir.glob("*.psd1"))

    issues: list[dict[str, Any]] = []
    for path in sorted(set(candidates)):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            values = [match.group(0) for match in WINDOWS_ABSOLUTE_PATH_RE.finditer(line)]
            values.extend(match.group(0) for match in POSIX_HOME_PATH_RE.finditer(line))
            for value in values:
                if not _looks_private_path(value):
                    continue
                issues.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "value": value,
                        "message": "User-specific absolute path should become an environment variable, config value, or relative path.",
                    }
                )
    return issues
