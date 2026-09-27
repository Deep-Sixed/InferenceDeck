from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

from .proc import run as run_hidden


def _run(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return run_hidden(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


DRAFT_MODEL_SUGGESTIONS = {
    "7b": ["Qwen2.5-1.5B-Instruct", "SmolLM2-360M", "Qwen2.5-3B-Instruct"],
    "8b": ["Qwen2.5-1.5B-Instruct", "SmolLM2-360M", "Qwen2.5-3B-Instruct"],
    "13b": ["Qwen2.5-1.5B-Instruct", "SmolLM2-360M", "Qwen2.5-3B-Instruct"],
    "14b": ["Qwen2.5-3B-Instruct", "SmolLM2-1.7B", "Qwen2.5-1.5B-Instruct"],
    "20b": ["Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"],
    "32b": ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"],
    "35b": ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"],
    "70b": ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"],
    "40b": ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"],
    "80b": ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"],
}


def _detect_base_model_size(model_name: str | None) -> str | None:
    if not model_name:
        return None
    text = model_name.lower()
    for size in ["80b", "70b", "40b", "35b", "32b", "20b", "14b", "13b", "8b", "7b", "3b", "1.5b", "1b"]:
        if f"{size}b" in text or f"{size}b" in text.replace(" ", ""):
            return size
    size_match = re.search(r"(\d+(?:\.\d+)?)\s*b", text)
    if size_match:
        value = float(size_match.group(1))
        if value >= 75:
            return "80b"
        if value >= 50:
            return "70b"
        if value >= 35:
            return "40b"
        if value >= 25:
            return "35b"
        if value >= 20:
            return "32b"
        if value >= 15:
            return "20b"
        if value >= 12:
            return "14b"
        if value >= 8:
            return "13b"
        if value >= 5:
            return "8b"
        if value >= 3:
            return "7b"
        if value >= 2:
            return "3b"
        if value >= 1:
            return "1.5b"
        return "1b"
    return None


def suggest_draft_models(base_model_name: str | None) -> list[dict[str, Any]]:
    size = _detect_base_model_size(base_model_name)
    if not size or size not in DRAFT_MODEL_SUGGESTIONS:
        return []
    suggestions = DRAFT_MODEL_SUGGESTIONS[size]
    results = []
    for name in suggestions:
        repo_id = f"Qwen/{name}-GGUF"
        verified = False
        try:
            import urllib.request
            search_url = f"https://huggingface.co/api/models/{repo_id}?limit=1"
            req = urllib.request.Request(search_url, headers={"User-Agent": "inferencedeck"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                verified = resp.status == 200
        except Exception:
            verified = False
        results.append({
            "repo_id": repo_id,
            "name": name,
            "description": f"{name} is a compatible draft model for speculative decoding with {size} base models.",
            "recommended_quant": "Q4_K_M",
            "verified": verified,
        })
    return results


def download_model_file(repo_id: str, filename: str, dest_dir: str) -> dict[str, Any]:
    """Re-download one file from a repo into dest_dir, overwriting in place.

    Used by the selected-model HF update flow: caller passes the exact repo and
    filename from a prior update check, so this is a targeted refresh, not a guess.
    """
    hf_cli = shutil.which("huggingface-cli")
    if not hf_cli:
        return {"success": False, "message": "Hugging Face CLI not found. Install with 'pip install huggingface_hub'."}
    if not (repo_id and filename and dest_dir):
        return {"success": False, "message": "repo_id, filename, and dest_dir are all required."}
    result = _run(
        [hf_cli, "download", repo_id, "--include", filename, "--local-dir", dest_dir],
        timeout=1800.0,
    )
    if result and result.returncode == 0:
        return {"success": True, "message": f"Downloaded {filename} from {repo_id} into {dest_dir}."}
    return {"success": False, "message": result.stderr.strip() if result else "Download failed."}


def pull_draft_model(repo_id: str, quant: str = "Q4_K_M") -> dict[str, Any]:
    hf_cli = shutil.which("huggingface-cli")
    if not hf_cli:
        return {"success": False, "message": "Hugging Face CLI not found. Install with 'pip install huggingface_hub'."}
    search_pattern = f"{repo_id}/{quant}"
    result = _run(
        [hf_cli, "ls", repo_id, "--patterns", f"*{quant}*"],
        timeout=15.0,
    )
    if not result or result.returncode != 0:
        result = _run(
            [hf_cli, "ls", repo_id],
            timeout=15.0,
        )
    if result and result.returncode == 0 and result.stdout.strip():
        files = result.stdout.strip().splitlines()
        gguf_files = [f for f in files if f.lower().endswith(".gguf")]
        if not gguf_files:
            return {"success": False, "message": "No GGUF files found for this repo/quant combination."}
        target_file = gguf_files[0]
    else:
        return {"success": False, "message": f"Could not list files for {repo_id}. Check the repo ID."}
    download_result = _run(
        [hf_cli, "download", repo_id, "--include", target_file],
        timeout=600.0,
    )
    if download_result and download_result.returncode == 0:
        return {"success": True, "message": f"Downloaded {target_file} from {repo_id}."}
    return {
        "success": False,
        "message": download_result.stderr.strip() if download_result else "Download failed.",
    }
