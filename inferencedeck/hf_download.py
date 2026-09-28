"""Pull GGUF models from Hugging Face by repo and quant or filename glob.

A model repo usually holds many quants, some split into ``-0000N-of-0000M``
shards, plus an optional vision projector (``mmproj``). Files are grouped into
*variants* (one quant = one file or one shard set) so a request like
``quant="Q4_K_M"`` or ``pattern="*Q4_K_M*"`` resolves to exactly one variant, all
of its shards, and a matching mmproj. No match or an ambiguous match fails with
the list of available variants instead of guessing.

Listing uses the public HF API; the download itself runs the ``hf`` (or older
``huggingface-cli``) CLI, which handles resume, auth and the HF cache that model
discovery already scans.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import PurePosixPath
from typing import Any

from .hf_metadata import HF_API, _headers
from .models import SPLIT_RE, parse_quant
from .proc import run as run_hidden

NEXT_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
MAX_TREE_PAGES = 50
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
DOWNLOAD_TIMEOUT_SECONDS = 6 * 3600
# Projector precision preference: full-ish precision first, it's small anyway.
MMPROJ_PREFERENCE = ("F16", "BF16", "F32", "Q8_0")


def valid_repo_id(repo_id: str | None) -> bool:
    return bool(repo_id) and bool(REPO_ID_RE.match(str(repo_id))) and ".." not in str(repo_id)


def _get_page(url: str) -> tuple[Any, str | None]:
    """One tree page and the next page's URL (the API paginates via ``Link``)."""
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        match = NEXT_LINK_RE.search(resp.headers.get("Link") or "")
        return json.loads(resp.read().decode("utf-8")), match.group(1) if match else None


def list_repo_files(repo_id: str, revision: str = "main") -> list[dict[str, Any]]:
    """Every file in a repo as ``{"path", "size"}`` (LFS size when present)."""
    quoted = urllib.parse.quote(repo_id, safe="/")
    url: str | None = f"{HF_API}/models/{quoted}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
    entries: list[Any] = []
    for _ in range(MAX_TREE_PAGES):
        if not url:
            break
        page, url = _get_page(url)
        entries.extend(page if isinstance(page, list) else [])
    files = []
    for entry in entries:
        if entry.get("type") != "file" or not entry.get("path"):
            continue
        lfs = entry.get("lfs") or {}
        files.append({"path": entry["path"], "size": lfs.get("size") or entry.get("size")})
    return files


def _is_mmproj(path: str) -> bool:
    return "mmproj" in PurePosixPath(path).name.lower()


def gguf_variants(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Group GGUF files into downloadable variants and list mmproj files."""
    variants: dict[str, dict[str, Any]] = {}
    mmproj: list[dict[str, Any]] = []
    for entry in files:
        path = str(entry["path"])
        if not path.lower().endswith(".gguf"):
            continue
        if _is_mmproj(path):
            mmproj.append({"path": path, "size_bytes": entry.get("size"), "quant": parse_quant(PurePosixPath(path).name)})
            continue
        pure = PurePosixPath(path)
        split = SPLIT_RE.match(pure.name)
        name = str(pure.parent / split.group("base")) if split else path
        if name.startswith("./"):
            name = name[2:]
        variant = variants.setdefault(name, {
            "name": name,
            "quant": parse_quant(pure.name),
            "files": [],
            "size_bytes": 0,
            "split_total": int(split.group("total")) if split else None,
        })
        variant["files"].append(path)
        if isinstance(entry.get("size"), int):
            variant["size_bytes"] += entry["size"]
    ordered = sorted(variants.values(), key=lambda v: v["name"].lower())
    for variant in ordered:
        variant["files"].sort()
        if variant["split_total"] and len(variant["files"]) != variant["split_total"]:
            variant["incomplete"] = True
    return {"variants": ordered, "mmproj": sorted(mmproj, key=lambda m: m["path"].lower())}


def _variant_matches(variant: dict[str, Any], pattern: str | None, quant: str | None) -> bool:
    if quant and str(variant.get("quant") or "").upper() != quant.upper():
        return False
    if pattern:
        glob = pattern.lower()
        candidates = [variant["name"], *variant["files"]]
        return any(fnmatch.fnmatch(item.lower(), glob) or fnmatch.fnmatch(PurePosixPath(item).name.lower(), glob) for item in candidates)
    return True


def _pick_mmproj(mmproj: list[dict[str, Any]], model_path: str) -> dict[str, Any] | None:
    if not mmproj:
        return None
    folder = str(PurePosixPath(model_path).parent)
    same_folder = [m for m in mmproj if str(PurePosixPath(m["path"]).parent) == folder] or mmproj

    def rank(item: dict[str, Any]) -> tuple[int, str]:
        quant = str(item.get("quant") or "").upper()
        return (MMPROJ_PREFERENCE.index(quant) if quant in MMPROJ_PREFERENCE else len(MMPROJ_PREFERENCE), item["path"])

    return min(same_folder, key=rank)


def _variant_summary(variant: dict[str, Any]) -> dict[str, Any]:
    return {key: variant.get(key) for key in ("name", "quant", "size_bytes", "split_total")} | {"file_count": len(variant["files"])}


def repo_gguf_listing(repo_id: str, revision: str = "main") -> dict[str, Any]:
    """The quants (variants) and projectors a repo offers."""
    if not valid_repo_id(repo_id):
        return {"success": False, "error": f"Invalid Hugging Face repo id: {repo_id!r} (expected owner/name)."}
    try:
        files = list_repo_files(repo_id, revision)
    except Exception as exc:
        return {"success": False, "repo_id": repo_id, "error": f"Could not list {repo_id}: {exc}"}
    grouped = gguf_variants(files)
    return {
        "success": True,
        "repo_id": repo_id,
        "revision": revision,
        "variants": [_variant_summary(v) | {"files": v["files"]} for v in grouped["variants"]],
        "mmproj": grouped["mmproj"],
    }


def resolve_download(
    repo_id: str,
    pattern: str | None = None,
    quant: str | None = None,
    include_mmproj: bool = True,
    revision: str = "main",
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pick exactly one variant (all shards) plus an optional projector."""
    if not valid_repo_id(repo_id):
        return {"success": False, "error": f"Invalid Hugging Face repo id: {repo_id!r} (expected owner/name)."}
    pattern = (pattern or "").strip() or None
    quant = (quant or "").strip() or None
    if files is None:
        try:
            files = list_repo_files(repo_id, revision)
        except Exception as exc:
            return {"success": False, "repo_id": repo_id, "error": f"Could not list {repo_id}: {exc}"}
    grouped = gguf_variants(files)
    variants = grouped["variants"]
    available = [_variant_summary(v) for v in variants]
    if not variants:
        return {"success": False, "repo_id": repo_id, "error": f"{repo_id} has no GGUF model files.", "available": available}

    matches = [v for v in variants if _variant_matches(v, pattern, quant)]
    wanted = " and ".join(part for part in (f"quant {quant}" if quant else "", f"pattern {pattern!r}" if pattern else "") if part) or "no filter"
    if not matches:
        return {"success": False, "repo_id": repo_id, "error": f"No GGUF in {repo_id} matches {wanted}.", "available": available}
    if len(matches) > 1:
        return {
            "success": False,
            "repo_id": repo_id,
            "error": f"{len(matches)} GGUF variants in {repo_id} match {wanted}; narrow it with a quant or filename pattern.",
            "matches": [_variant_summary(v) for v in matches],
            "available": available,
        }

    variant = matches[0]
    warnings: list[str] = []
    if variant.get("incomplete"):
        warnings.append(f"Repo lists {len(variant['files'])} of {variant['split_total']} shards for {variant['name']}; the model may not load.")
    projector = _pick_mmproj(grouped["mmproj"], variant["files"][0]) if include_mmproj else None
    download_files = list(variant["files"]) + ([projector["path"]] if projector else [])
    total = variant["size_bytes"] + ((projector.get("size_bytes") or 0) if projector else 0)
    return {
        "success": True,
        "repo_id": repo_id,
        "revision": revision,
        "variant": _variant_summary(variant),
        "mmproj": projector["path"] if projector else None,
        "files": download_files,
        "total_bytes": total,
        "warnings": warnings,
    }


def find_hf_cli() -> str | None:
    """``hf`` (current CLI) or ``huggingface-cli`` (older name)."""
    return shutil.which("hf") or shutil.which("huggingface-cli")


def download_model(
    repo_id: str,
    pattern: str | None = None,
    quant: str | None = None,
    include_mmproj: bool = True,
    dest_dir: str | None = None,
    revision: str = "main",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve then download one variant. Without ``dest_dir`` files land in the HF cache."""
    plan = resolve_download(repo_id, pattern=pattern, quant=quant, include_mmproj=include_mmproj, revision=revision)
    if not plan.get("success"):
        return plan
    if dry_run:
        return plan | {"dry_run": True}
    cli = find_hf_cli()
    if not cli:
        return plan | {"success": False, "error": "Hugging Face CLI not found. Install it with 'pip install huggingface_hub'."}
    if any(path.startswith("-") for path in plan["files"]):
        return plan | {"success": False, "error": "Refusing a repo file name that starts with '-'."}
    argv = [cli, "download", repo_id, *plan["files"], "--revision", revision]
    if dest_dir:
        argv.extend(["--local-dir", str(dest_dir)])
    try:
        result = run_hidden(argv, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return plan | {"success": False, "error": "Download timed out; run it again to resume."}
    except OSError as exc:
        return plan | {"success": False, "error": f"Could not run {cli}: {exc}"}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return plan | {"success": False, "error": detail[-1] if detail else f"{cli} exited with {result.returncode}."}
    output = (result.stdout or "").strip().splitlines()
    location = str(dest_dir) if dest_dir else (output[-1].strip() if output else None)
    return plan | {"location": location, "message": f"Downloaded {len(plan['files'])} file(s) from {repo_id}."}
