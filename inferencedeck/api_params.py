"""Validation for launch settings accepted over the HTTP control API.

Python callers of ControlPlane may pass any profile override. The HTTP API is
reachable by browsers and trays, so it only accepts these tuning keys with
bounded values. Anything that picks what runs or where it listens (host, port,
binary/model/draft paths, runtime, device, tensor overrides, extra args) stays
in the profile or app config and cannot be changed remotely.
"""

from __future__ import annotations

from typing import Any

from .sampling import NO_PRESET, SAMPLING_PRESETS

CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}

# key -> (kind, low, high); kind is "int", "float", "bool", "cache" or "preset"
ALLOWED_OVERRIDES: dict[str, tuple[str, float, float]] = {
    "ctx_size": ("int", 512, 1_048_576),
    "gpu_layers": ("int", 0, 999),
    "threads": ("int", 1, 512),
    "threads_batch": ("int", 1, 512),
    "batch_size": ("int", 1, 65_536),
    "ubatch_size": ("int", 1, 65_536),
    "n_predict": ("int", -1, 1_000_000),
    "seed": ("int", -1, 2**32 - 1),
    "top_k": ("int", 0, 1_000),
    "temperature": ("float", 0, 5),
    "top_p": ("float", 0, 1),
    "min_p": ("float", 0, 1),
    "repeat_penalty": ("float", 0, 5),
    "presence_penalty": ("float", -2, 2),
    "frequency_penalty": ("float", -2, 2),
    "flash_attn": ("bool", 0, 0),
    "jinja": ("bool", 0, 0),
    "reasoning": ("bool", 0, 0),
    "cache_type_k": ("cache", 0, 0),
    "cache_type_v": ("cache", 0, 0),
    # Not a llama-server flag: InferenceDeck's idle auto-release window (0 = off).
    "idle_release_seconds": ("int", 0, 7 * 24 * 3600),
    "repeat_last_n": ("int", -1, 1_000_000),
    # A sampling preset from sampling.py (or "none"); explicit overrides still win.
    "sampling_preset": ("preset", 0, 0),
    # Pass the projector found next to the model (--mmproj) for image/audio input.
    "vision": ("bool", 0, 0),
}


def validate_overrides(raw: Any) -> dict[str, Any] | None:
    """Return a clean overrides dict, or raise ValueError naming the first bad key."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("overrides must be an object")
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        spec = ALLOWED_OVERRIDES.get(key)
        if spec is None:
            raise ValueError(f"override '{key}' is not allowed over the API")
        kind, low, high = spec
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif kind == "preset":
            if value != NO_PRESET and value not in SAMPLING_PRESETS:
                raise ValueError(f"{key} must be one of {', '.join([NO_PRESET, *SAMPLING_PRESETS])}")
        elif kind == "cache":
            if value not in CACHE_TYPES:
                raise ValueError(f"{key} must be one of {', '.join(sorted(CACHE_TYPES))}")
        else:
            # bool is an int subclass; reject it explicitly.
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not numeric or (kind == "int" and not float(value).is_integer()):
                raise ValueError(f"{key} must be a{'n integer' if kind == 'int' else ' number'}")
            if not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            value = int(value) if kind == "int" else float(value)
        clean[key] = value
    return clean
