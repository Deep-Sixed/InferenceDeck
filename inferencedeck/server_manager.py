from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import detect_llama_cpp, detect_runtime
from .config import AppConfig
from .llama_args import LaunchCommand, build_llama_server_args
from .paths import cache_dir, find_project_root, is_windows
from .profile_resolver import ResolvedProfile, resolve_profiles


STATE_FILENAME = "servers.json"
LOG_DIRNAME = "logs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path() -> Path:
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / STATE_FILENAME


def log_dir() -> Path:
    path = cache_dir() / LOG_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return {"servers": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"servers": []}


def write_state(state: dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    if is_windows():
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return str(int(pid)) in result.stdout
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    # A killed-but-unreaped child becomes a zombie; os.kill(pid, 0) still
    # succeeds for it. Treat zombies as dead so Stop reports success and the
    # server list doesn't show a corpse as running. Linux /proc only; elsewhere
    # we keep the os.kill result.
    # ponytail: /proc check, fine until this needs to run on macOS.
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="ascii") as f:
            return f.read().rpartition(")")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _wait_gone(pid: int, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not pid_is_running(pid):
            return True
        time.sleep(0.25)
    return False


def tail_file(path: str | Path | None, lines: int = 120) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError as exc:
        return f"Could not read {file_path}: {exc}"


def list_servers() -> list[dict[str, Any]]:
    prune_stale_servers()
    state = read_state()
    servers = []
    for server in state.get("servers", []):
        item = dict(server)
        item["running"] = pid_is_running(item.get("pid"))
        servers.append(item)
    return servers


def prune_stale_servers() -> None:
    """Remove entries for PIDs that are no longer running."""
    state = read_state()
    servers = state.get("servers", [])
    kept = [s for s in servers if pid_is_running(s.get("pid")) or not s.get("pid")]
    if len(kept) != len(servers):
        state["servers"] = kept
        write_state(state)


def trim_server_history(limit: int = 5) -> None:
    state = read_state()
    servers = state.get("servers", [])
    if len(servers) <= limit:
        return
    kept = servers[:limit]
    state["servers"] = kept
    write_state(state)


def _find_server(server_id: str | None = None, mode: str | None = None) -> dict[str, Any] | None:
    for server in list_servers():
        if server_id and server.get("id") == server_id:
            return server
        if mode and server.get("mode") == mode:
            return server
    return None


def stop_server(server_id: str | None = None, mode: str | None = None, timeout: int = 10) -> dict[str, Any]:
    server = _find_server(server_id, mode)
    if not server:
        return {"success": True, "message": "No tracked server matched the request."}

    raw_pid = server.get("pid")
    if not raw_pid:
        return {"success": True, "message": "Tracked server has no PID to stop."}
    pid = int(raw_pid)
    if not pid_is_running(pid):
        _update_server(server["id"], {"status": "stopped", "running": False, "stopped_at": _now()})
        return {"success": True, "message": f"Tracked PID {pid} is no longer running."}

    if is_windows():
        cmd = ["taskkill", "/PID", str(pid), "/T", "/F"]
    else:
        cmd = ["kill", str(pid)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        still_running = pid_is_running(pid)
        _update_server(
            server["id"],
            {
                "status": "stop_failed",
                "running": still_running,
                "stopped_at": _now(),
                "stop_stdout": "",
                "stop_stderr": str(exc),
            },
        )
        return {
            "success": False,
            "message": str(exc),
            "server": _find_server(server["id"]),
        }

    if result.returncode != 0:
        still_running = pid_is_running(pid)
        _update_server(
            server["id"],
            {
                "status": "stop_failed",
                "running": still_running,
                "stopped_at": _now(),
                "stop_stdout": result.stdout.strip(),
                "stop_stderr": result.stderr.strip(),
            },
        )
        return {
            "success": False,
            "message": result.stdout.strip() or result.stderr.strip() or f"taskkill returned {result.returncode} for PID {pid}.",
            "server": _find_server(server["id"]),
        }

    def _stopped_ok(message: str) -> dict[str, Any]:
        _update_server(
            server["id"],
            {
                "status": "stopped",
                "running": False,
                "stopped_at": _now(),
                "stop_stdout": result.stdout.strip(),
                "stop_stderr": result.stderr.strip(),
            },
        )
        return {"success": True, "message": message, "server": _find_server(server["id"])}

    if _wait_gone(pid, 5):
        return _stopped_ok(result.stdout.strip() or result.stderr.strip() or f"Stopped PID {pid}.")

    # SIGTERM was ignored. Windows taskkill already forced (/F); on POSIX
    # escalate to SIGKILL so the Stop button can't be defeated by a hung server.
    if not is_windows():
        subprocess.run(["kill", "-9", str(pid)], capture_output=True, text=True, timeout=timeout, check=False)
        if _wait_gone(pid, 3):
            return _stopped_ok(f"Stopped PID {pid} with SIGKILL after it ignored SIGTERM.")

    _update_server(
        server["id"],
        {
            "status": "stop_failed",
            "running": True,
            "stopped_at": _now(),
            "stop_stdout": result.stdout.strip(),
            "stop_stderr": result.stderr.strip(),
        },
    )
    return {
        "success": False,
        "message": f"PID {pid} did not exit after SIGTERM and SIGKILL.",
        "server": _find_server(server["id"]),
    }


def _update_server(server_id: str, patch: dict[str, Any]) -> None:
    state = read_state()
    servers = state.setdefault("servers", [])
    for idx, server in enumerate(servers):
        if server.get("id") == server_id:
            servers[idx] = {**server, **patch}
            write_state(state)
            return


def _upsert_server(server: dict[str, Any]) -> None:
    state = read_state()
    servers = state.setdefault("servers", [])
    for idx, existing in enumerate(servers):
        if existing.get("id") == server.get("id"):
            servers[idx] = server
            write_state(state)
            return
    servers.append(server)
    write_state(state)


def _health_url(host: str, port: int) -> str:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"http://{probe_host}:{int(port)}/v1/models"


def wait_until_ready(host: str, port: int, pid: int, timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    url = _health_url(host, port)
    while time.time() < deadline:
        if not pid_is_running(pid):
            return False
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(1)
    return False


def _profile_by_mode(mode: str, project_root: str | Path | None, model_dirs: list[str | Path] | None) -> ResolvedProfile | None:
    for profile in resolve_profiles(project_root=project_root, model_dirs=model_dirs):
        if profile.mode == mode:
            return profile
    return None


def prepare_launch_command(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve() if project_root else find_project_root()
    app_config = config or AppConfig.load()
    resolved = _profile_by_mode(mode, root, model_dirs or app_config.model_dirs)
    if not resolved:
        return {"success": False, "error": f"Unknown profile mode: {mode}"}
    if not resolved.launchable or not resolved.model:
        return {
            "success": False,
            "error": "Profile is not launchable.",
            "profile": resolved.to_dict(),
        }

    params = dict(resolved.params)
    params.update(overrides or {})
    params.setdefault("host", app_config.default_host)
    params.setdefault("port", app_config.default_port)

    runtime = str(params.get("runtime") or "llama.cpp").strip() or "llama.cpp"
    if runtime != "llama.cpp":
        env = detect_runtime(runtime, root, config=app_config)
        if env is None:
            return {"success": False, "error": f"Unknown runtime: {runtime}"}
        # llama.cpp is the only runtime wired into the launch path so far; report a
        # clear error for the others instead of silently starting llama.cpp.
        return {
            "success": False,
            "error": (
                f"{env.name} is selected but cannot be launched from here yet — "
                "only llama.cpp is wired into Start/Fit. Switch the Runtime back to "
                "llama.cpp to launch this profile."
            ),
            "environment": env.to_dict(),
            "profile": resolved.to_dict(),
        }

    llama = detect_llama_cpp(root, config=app_config)
    if not llama.binary_path:
        return {"success": False, "error": "llama-server was not found.", "environment": llama.to_dict()}

    command = build_llama_server_args(
        llama.binary_path,
        resolved.model["path"],
        params,
        extra_args=app_config.extra_llama_args,
    )
    warnings = resolved.warnings + command.warnings
    return {
        "success": True,
        "profile": resolved.to_dict(),
        "environment": llama.to_dict(),
        "command": command.to_dict(),
        "params": params,
        "warnings": warnings,
    }


def start_profile(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    stop_existing: bool = False,
    wait_ready: bool = True,
    ready_timeout_seconds: int = 45,
) -> dict[str, Any]:
    prepared = prepare_launch_command(mode, project_root, model_dirs, overrides)
    if not prepared.get("success"):
        return prepared

    existing = _find_server(mode=mode)
    if existing and existing.get("running"):
        if not stop_existing:
            return {
                "success": False,
                "error": f"Profile '{mode}' already has a tracked running server.",
                "server": existing,
            }
        stop_result = stop_server(mode=mode)
        if not stop_result.get("success"):
            return {"success": False, "error": "Could not stop existing tracked server.", "stop_result": stop_result}

    command = LaunchCommand(
        argv=prepared["command"]["argv"],
        cwd=prepared["command"]["cwd"],
        warnings=prepared["command"].get("warnings", []),
    )
    mode_slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in mode).strip("-") or "server"
    stdout_path = log_dir() / f"{mode_slug}-stdout.log"
    stderr_path = log_dir() / f"{mode_slug}-stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace")
    stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            shell=False,
            # Detach so the managed server outlives the control center: a
            # terminal/process-group signal (Ctrl-C, systemd stop) to us must
            # not take down the servers we track in state. setsid on POSIX,
            # ignored on Windows (CREATE_NO_WINDOW already detaches the console).
            start_new_session=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        stdout_handle.close()
        stderr_handle.close()
        return {"success": False, "error": str(exc), "prepared": prepared}
    finally:
        stdout_handle.close()
        stderr_handle.close()

    params = prepared["params"]
    server_id = f"{mode}-{proc.pid}"
    server = {
        "id": server_id,
        "mode": mode,
        "pid": proc.pid,
        "status": "starting",
        "running": True,
        "host": str(params.get("host", "127.0.0.1")),
        "port": int(params.get("port", 8080)),
        "model_path": prepared["profile"]["model"]["path"] if prepared["profile"].get("model") else None,
        "command_line": command.command_line,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "started_at": _now(),
        "warnings": prepared.get("warnings", []),
    }
    _upsert_server(server)
    app_config = AppConfig.load()
    trim_server_history(app_config.server_history_limit)

    if wait_ready:
        ready = wait_until_ready(server["host"], server["port"], proc.pid, ready_timeout_seconds)
        _update_server(
            server_id,
            {
                "status": "running" if ready else "startup_timeout",
                "running": pid_is_running(proc.pid),
                "ready_at": _now() if ready else None,
            },
        )
        if not ready:
            return {
                "success": False,
                "error": "Server process started but did not become ready before timeout.",
                "server": _find_server(server_id),
                "stderr_tail": tail_file(stderr_path),
            }

    return {"success": True, "server": _find_server(server_id), "prepared": prepared}


def server_logs(server_id: str, lines: int = 200) -> dict[str, Any]:
    server = _find_server(server_id)
    if not server:
        return {"success": False, "error": f"Unknown tracked server: {server_id}"}
    return {
        "success": True,
        "server": server,
        "stdout": tail_file(server.get("stdout_log"), lines),
        "stderr": tail_file(server.get("stderr_log"), lines),
    }
