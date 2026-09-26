"""Best-effort runtime update checks against public release sources.

The control center never downloads or replaces binaries on its own. This
module only compares the locally detected version of each runtime against
the latest release tag published by the upstream project, caches the
result for a short window so refreshes stay cheap, and returns a small
payload that the UI can render as an "Update available" badge with a link
to the release page.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .paths import cache_dir


CACHE_FILENAME = "runtime-updates.json"
DEFAULT_CACHE_TTL_SECONDS = 60 * 60  # one hour
REQUEST_TIMEOUT_SECONDS = 4.0
USER_AGENT = "inferencedeck/runtime-updates"


SUPPORTED_CHANNELS = ("stable", "prerelease")


@dataclass
class RuntimeUpdateInfo:
    """One runtime's update availability result."""

    runtime_id: str
    runtime_name: str
    current_version: str | None
    latest_version: str | None
    update_available: bool
    channel: str
    release_url: str | None = None
    checked_at: float = 0.0
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Maps a runtime environment id (from inferencedeck.schema.Environment) to the
# GitHub repository that publishes its releases. Only runtimes with a
# reliable public release source are listed; everything else is skipped.
GITHUB_REPOS: dict[str, str] = {
    "llama.cpp": "ggml-org/llama.cpp",
    "wsl-llama.cpp": "ggml-org/llama.cpp",
    "ollama": "ollama/ollama",
    "vllm": "vllm-project/vllm",
    "mlx": "ml-explore/mlx",
}


def runtime_label(runtime_id: str) -> str:
    return {
        "llama.cpp": "llama.cpp",
        "wsl-llama.cpp": "llama.cpp (WSL)",
        "ollama": "Ollama",
        "vllm": "vLLM",
        "mlx": "MLX",
        "lm-studio": "LM Studio",
    }.get(runtime_id, runtime_id)


def _strip_leading_v(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned[:1].lower() == "v":
        cleaned = cleaned[1:]
    return cleaned


def parse_version(value: str | None) -> tuple[int, ...] | None:
    """Parse a version string into a comparable tuple.

    Handles three common shapes used by the runtimes we track:

    - semver: ``1.2.3`` or ``1.2.3-rc1`` (pre-release suffix is ignored)
    - llama.cpp build number: ``b4500`` or ``4500``
    - any leading or trailing bare integer: ``20240101``

    Returns None when no numeric component can be extracted. Pre-release
    suffixes are ignored for the comparison itself; the caller decides
    whether to consider pre-releases via the channel argument.
    """

    cleaned = _strip_leading_v(value)
    if cleaned is None:
        return None
    semver = re.match(r"^(\d+(?:\.\d+)*)", cleaned)
    if semver:
        return tuple(int(part) for part in semver.group(1).split(".") if part)
    build = re.match(r"^[a-z]+(\d+)$", cleaned.lower())
    if build:
        return (int(build.group(1)),)
    bare = re.search(r"(\d+)$", cleaned)
    if bare:
        return (int(bare.group(1)),)
    return None


def is_prerelease_tag(tag: str | None) -> bool:
    cleaned = (_strip_leading_v(tag) or "").lower()
    if not cleaned:
        return False
    return any(marker in cleaned for marker in ("-rc", "-pre", "-alpha", "-beta", "-dev", "preview"))


def compare_versions(current: str | None, latest: str | None) -> int:
    """Return -1/0/1 comparing current to latest. Unknown -> 0."""

    cur = parse_version(current)
    nxt = parse_version(latest)
    if cur is None or nxt is None:
        return 0
    if cur < nxt:
        return -1
    if cur > nxt:
        return 1
    return 0


def github_release_api_url(repo: str, channel: str) -> str:
    if channel == "prerelease":
        return f"https://api.github.com/repos/{repo}/releases"
    return f"https://api.github.com/repos/{repo}/releases/latest"


def github_release_html_url(repo: str, tag: str | None) -> str:
    if tag:
        return f"https://github.com/{repo}/releases/tag/{tag}"
    return f"https://github.com/{repo}/releases"


def _request_json(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> tuple[bool, Any, str | None]:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return True, json.loads(raw.decode("utf-8")), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, None, str(exc)


def _extract_tag(payload: Any, channel: str) -> str | None:
    if not payload:
        return None
    if channel == "prerelease":
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first.get("tag_name") or first.get("name")
        return None
    if isinstance(payload, dict):
        return payload.get("tag_name") or payload.get("name")
    return None


def fetch_latest_release(repo: str, channel: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch the latest release tag for a GitHub repo.

    Returns a dict with ``tag``, ``release_url``, ``ok``, and ``error`` keys.
    Never raises: network errors surface as ``ok=False`` with a message.
    """

    if channel not in SUPPORTED_CHANNELS:
        channel = "stable"
    url = github_release_api_url(repo, channel)
    ok, payload, error = _request_json(url, timeout=timeout)
    if not ok:
        return {
            "ok": False,
            "tag": None,
            "release_url": github_release_html_url(repo, None),
            "error": error,
        }
    tag = _extract_tag(payload, channel)
    if channel == "prerelease" and tag and not is_prerelease_tag(tag):
        tag = None
    return {
        "ok": True,
        "tag": tag,
        "release_url": github_release_html_url(repo, tag),
        "error": None,
    }


def _cache_path() -> Path:
    return cache_dir() / CACHE_FILENAME


def load_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(data: dict[str, Any]) -> Path:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _cache_fresh(entry: dict[str, Any] | None, ttl_seconds: int) -> bool:
    if not entry:
        return False
    checked_at = entry.get("checked_at") or 0
    if not isinstance(checked_at, (int, float)):
        return False
    return (time.time() - checked_at) <= ttl_seconds


def _build_info(
    runtime_id: str,
    current_version: str,
    fetched: dict[str, Any],
    channel: str,
    now: float,
) -> RuntimeUpdateInfo:
    latest = fetched.get("tag") if fetched.get("ok") else None
    update_available = False
    if latest:
        update_available = compare_versions(current_version, latest) < 0
    return RuntimeUpdateInfo(
        runtime_id=runtime_id,
        runtime_name=runtime_label(runtime_id),
        current_version=current_version,
        latest_version=latest,
        update_available=update_available,
        channel=channel,
        release_url=fetched.get("release_url"),
        checked_at=now,
        notes=fetched.get("error"),
    )


def _candidate_runtimes(environments: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return (runtime_id, current_version) pairs worth checking.

    A runtime is worth checking when it has an upstream GitHub repo AND
    a detected installed version. Runtimes that are not available locally
    or have no version to compare cannot be evaluated and are skipped.
    """

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for env in environments or []:
        runtime_id = env.get("id") or env.get("kind")
        if not runtime_id or runtime_id in seen:
            continue
        if runtime_id not in GITHUB_REPOS:
            continue
        seen.add(runtime_id)
        version = env.get("version")
        if not version:
            details = env.get("details") or {}
            version = (
                details.get("version")
                or details.get("tag")
                or details.get("installed_version")
            )
        if not version:
            continue
        candidates.append((runtime_id, str(version)))
    return candidates


def check_runtime_updates(
    environments: Iterable[dict[str, Any]] | None,
    channel: str = "stable",
    *,
    force_refresh: bool = False,
    force_runtime: str | None = None,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Compute update availability for each supported runtime.

    Returns a dict with ``channel``, ``checked_at``, ``updates`` (list of
    RuntimeUpdateInfo as dicts), and ``cache_path``. Network failures are
    captured per-runtime in ``notes`` rather than raised.
    """

    if channel not in SUPPORTED_CHANNELS:
        channel = "stable"

    cache = load_cache() if not force_refresh else {}
    cache_key = f"{channel}"
    cached_block = cache.get(cache_key) if isinstance(cache, dict) else None

    candidates = _candidate_runtimes(environments or [])
    now = time.time()

    results: list[RuntimeUpdateInfo] = []
    for runtime_id, current_version in candidates:
        repo = GITHUB_REPOS[runtime_id]
        cached_entry: dict[str, Any] | None = None
        if cached_block and isinstance(cached_block, dict):
            entry = cached_block.get(runtime_id)
            if isinstance(entry, dict):
                cached_entry = entry

        fetched: dict[str, Any]
        # force_runtime bypasses the cache for just one runtime (per-card recheck)
        # while the rest still serve from cache, so a single recheck makes one
        # GitHub call instead of N.
        if runtime_id != force_runtime and cached_entry and _cache_fresh(cached_entry, ttl_seconds):
            fetched = {
                "ok": True,
                "tag": cached_entry.get("latest_version"),
                "release_url": cached_entry.get("release_url") or github_release_html_url(repo, None),
                "error": None,
            }
            checked_at = cached_entry.get("checked_at") or now
        else:
            fetched = fetch_latest_release(repo, channel, timeout=timeout_seconds)
            checked_at = now
            if fetched.get("ok"):
                cached_entry = {
                    "latest_version": fetched.get("tag"),
                    "release_url": fetched.get("release_url"),
                    "checked_at": checked_at,
                }
                cache.setdefault(cache_key, {})[runtime_id] = cached_entry

        info = _build_info(runtime_id, current_version, fetched, channel, checked_at)
        results.append(info)

    try:
        save_cache(cache)
    except OSError:
        pass

    all_runtime_ids = [env.get("id") or env.get("kind") for env in (environments or []) if env.get("id") or env.get("kind")]
    supported_but_no_version = [rid for rid in all_runtime_ids if rid in GITHUB_REPOS and not any(rid == cid for cid, _ in candidates)]
    unsupported_runtimes = [rid for rid in all_runtime_ids if rid not in GITHUB_REPOS]

    return {
        "channel": channel,
        "checked_at": now,
        "updates": [info.to_dict() for info in results],
        "supported_channels": list(SUPPORTED_CHANNELS),
        "known_runtime_count": len(all_runtime_ids),
        "checked_runtime_count": len(results),
        "skipped_no_version": supported_but_no_version,
        "skipped_unsupported": unsupported_runtimes,
    }
