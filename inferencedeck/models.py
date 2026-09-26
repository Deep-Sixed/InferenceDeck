from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from .paths import default_model_dirs, dedupe_paths
from .schema import ModelFile


QUANT_RE = re.compile(
    r"(?i)(?:^|[-_.])((?:i?q\d(?:_[a-z0-9]+)+)|(?:f16|bf16|f32))(?:[-_.]|$)"
)
PARAM_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*([bm])(?:[-_\s]|$)")
SPLIT_RE = re.compile(r"(?i)^(?P<base>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$")
GENERIC_MODEL_DIR_NAMES = {"gguf", "model", "models", "mtp", "weights"}


def _model_id(path: Path) -> str:
    raw = str(path).encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:16]


def parse_quant(filename: str) -> str | None:
    match = QUANT_RE.search(filename)
    if not match:
        return None
    return match.group(1).upper()


def parse_params(text: str) -> float | None:
    match = PARAM_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).lower()
    return value if suffix == "b" else value / 1000.0


def _find_mmproj(directory: Path) -> str | None:
    if not directory.is_dir():
        return None
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() == ".gguf" and "mmproj" in child.name.lower():
            return str(child)
    return None


def _split_info(path: Path) -> tuple[bool, int | None, list[Path]]:
    match = SPLIT_RE.match(path.name)
    if not match:
        return False, None, [path]
    part = int(match.group("part"))
    total = int(match.group("total"))
    pattern = f"{match.group('base')}-*-of-{match.group('total')}.gguf"
    parts = sorted(path.parent.glob(pattern))
    return part != 1, total, parts or [path]


def _source_name(path: Path, roots: list[tuple[str, Path]]) -> str:
    for source, root in roots:
        try:
            path.resolve().relative_to(root.resolve())
            return source
        except (OSError, ValueError):
            continue
    return "custom"


def discover_models(
    model_dirs: list[Path] | None = None,
    project_root: Path | None = None,
    max_files: int = 10000,
) -> list[ModelFile]:
    """Discover local model files without scanning broad user directories."""

    if model_dirs is None:
        roots = default_model_dirs(project_root)
    else:
        roots = [("custom", path) for path in dedupe_paths(model_dirs) if path.is_dir()]

    discovered: list[ModelFile] = []
    seen: set[str] = set()
    visited = 0

    for _, root in roots:
        if not root.is_dir():
            continue
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".git")]
            for filename in filenames:
                if not filename.lower().endswith(".gguf"):
                    continue
                path = Path(current_root) / filename
                if "mmproj" in filename.lower():
                    continue
                skip_split_part, split_total, split_parts = _split_info(path)
                if skip_split_part:
                    continue
                try:
                    key = str(path.resolve())
                except OSError:
                    key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                visited += 1
                if visited > max_files:
                    return sorted(discovered, key=lambda item: item.name.lower())

                parent = path.parent
                size = sum(part.stat().st_size for part in split_parts if part.exists())
                file_stem = path.stem
                if split_total:
                    split_match = SPLIT_RE.match(path.name)
                    file_stem = split_match.group("base") if split_match else file_stem
                if parent == root or parent.name.lower() in GENERIC_MODEL_DIR_NAMES:
                    name = file_stem
                else:
                    name = parent.name
                source = _source_name(path, roots)
                param_hint = parse_params(f"{parent.name} {file_stem}")
                warnings: list[str] = []
                if split_total and len(split_parts) < split_total:
                    warnings.append(f"Split GGUF appears incomplete: found {len(split_parts)} of {split_total} parts.")

                discovered.append(
                    ModelFile(
                        id=_model_id(path),
                        name=name,
                        path=str(path),
                        source=source,
                        format="GGUF",
                        size_bytes=size,
                        quant=parse_quant(path.name),
                        params_b=param_hint,
                        mmproj_path=_find_mmproj(parent),
                        split_total=split_total,
                        details={"primary_file": path.name, "root": str(root)},
                        warnings=warnings,
                    )
                )

    return sorted(discovered, key=lambda item: item.name.lower())
