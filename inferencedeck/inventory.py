from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import detect_all
from .config import AppConfig
from .manifest import load_profiles
from .models import discover_models
from .paths import default_model_dirs, find_project_root
from .portability import scan_portability_issues


def build_inventory(
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    include_manifest: bool = True,
    max_files: int = 10000,
) -> dict[str, Any]:
    """Build a portable inventory of runtimes, local models, and profiles."""

    root = Path(project_root).expanduser().resolve() if project_root else find_project_root()
    parsed_model_dirs = [Path(path).expanduser() for path in model_dirs] if model_dirs else None
    app_config = AppConfig.load()

    environments = detect_all(root, config=app_config)
    models = discover_models(parsed_model_dirs, root, max_files=max_files)
    profiles = load_profiles(root) if include_manifest else []
    portability_issues = scan_portability_issues(root)
    scan_roots = parsed_model_dirs or [path for _, path in default_model_dirs(root)]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root) if root else None,
        "scan_roots": [str(path) for path in scan_roots],
        "summary": {
            "environment_count": len(environments),
            "available_environment_count": len([env for env in environments if env.available]),
            "model_count": len(models),
            "profile_count": len(profiles),
            "profile_portability_warning_count": sum(len(profile.portable_warnings) for profile in profiles),
            "legacy_portability_issue_count": len(portability_issues),
        },
        "environments": [env.to_dict() for env in environments],
        "models": [model.to_dict() for model in models],
        "profiles": [profile.to_dict() for profile in profiles],
        "portability_issues": portability_issues,
    }
