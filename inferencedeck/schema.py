from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Environment:
    """A detected local inference environment or runtime."""

    id: str
    kind: str
    name: str
    available: bool
    binary_path: str | None = None
    api_url: str | None = None
    version: str | None = None
    model_count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelFile:
    """A locally discoverable model artifact."""

    id: str
    name: str
    path: str
    source: str
    format: str
    size_bytes: int
    quant: str | None = None
    params_b: float | None = None
    mmproj_path: str | None = None
    split_total: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelProfile:
    """A runnable profile loaded from a manifest such as models.json."""

    mode: str
    name: str
    description: str
    script: str | None
    script_path: str | None
    script_exists: bool
    model_path: str | None = None
    model_exists: bool | None = None
    model_size_gb: float | None = None
    recommended_params: dict[str, Any] = field(default_factory=dict)
    portable_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

