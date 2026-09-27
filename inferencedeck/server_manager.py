from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import LAUNCHABLE_RUNTIMES, detect_llama_cpp, detect_runtime, detect_vllm_cpp
from .config import AppConfig
from .llama_args import LaunchCommand, build_llama_server_args
from .vllm_cpp_args import build_vllm_cpp_server_args
from .paths import cache_dir, find_project_root, is_windows
from .profile_resolver import ResolvedProfile, resolve_profiles
from .proc import run as run_hidden
from .sampling import layer_sampling_preset, request_defaults


STATE_FILENAME = "servers.json"
# A parked server was stopped to free its GPU memory; its record keeps the
# restart spec (mode + overrides) so Restore can bring it back.
PARKED = "parked"
RESTORING = "restoring"
CONTEXT_PRESETS = (8192, 16384, 32768, 65536, 131072)
LOG_DIRNAME = "logs"
# Large GGUFs on slow disks routinely take over a minute to load; a short
# deadline marks healthy servers as startup_timeout.
DEFAULT_READY_TIMEOUT_SECONDS = 120
MIN_READY_TIMEOUT_SECONDS = 15
# llama-server /props modality keys mapped to input modalities.
_PROPS_MODALITIES = (("vision", "image"), ("audio", "audio"), ("video", "video"))


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
    # A unique temp name per write, so concurrent writers never share (and
    # clobber) one staging file.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state, indent=2) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# Guards read-modify-write of servers.json. The RLock covers threads in this
# process; the file lock covers other processes (e.g. the CLI next to the daemon).
_STATE_LOCK = threading.RLock()
_STATE_DEPTH = threading.local()


def _lock_file(fh: Any) -> None:
    if is_windows():
        import msvcrt

        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.05)
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)


def _unlock_file(fh: Any) -> None:
    if is_windows():
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def state_lock() -> Iterator[None]:
    with _STATE_LOCK:
        depth = getattr(_STATE_DEPTH, "value", 0)
        _STATE_DEPTH.value = depth + 1
        try:
            if depth:
                # Re-entered from this thread: the file lock is already held.
                yield
                return
            lock_path = state_path().with_suffix(".lock")
            with open(lock_path, "a+b") as fh:
                _lock_file(fh)
                try:
                    yield
                finally:
                    _unlock_file(fh)
        finally:
            _STATE_DEPTH.value = depth


def _mutate_state(change: Callable[[dict[str, Any]], bool | None]) -> None:
    """Apply ``change`` to the current state under the lock; write unless it returns False."""
    with state_lock():
        state = read_state()
        state.setdefault("servers", [])
        if change(state) is not False:
            write_state(state)


def _windows_pid_alive(pid: int) -> bool:
    """Ask the kernel directly; the status poll runs often, so no tasklist.exe per check."""
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Exists but belongs to someone we can't query (e.g. another user).
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    if is_windows():
        return _windows_pid_alive(int(pid))
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    # A killed-but-unreaped child becomes a zombie; os.kill(pid, 0) still
    # succeeds for it. Treat zombies as dead so Stop reports success and the
    # server list doesn't show a corpse as running. Linux exposes state in
    # /proc; macOS/BSD have no /proc, so ask ps instead.
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="ascii") as f:
            return f.read().rpartition(")")[2].split()[0] != "Z"
    except FileNotFoundError:
        pass
    except (OSError, IndexError):
        return True
    try:
        result = run_hidden(
            ["ps", "-o", "stat=", "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    stat = result.stdout.strip()
    if result.returncode != 0 and not stat:
        return False
    return not stat.startswith("Z")


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

    def change(state: dict[str, Any]) -> bool:
        servers = state["servers"]
        kept = [s for s in servers if pid_is_running(s.get("pid")) or not s.get("pid")]
        state["servers"] = kept
        return len(kept) != len(servers)

    _mutate_state(change)


def trim_server_history(limit: int = 5) -> None:
    """Cap non-running records at ``limit``, keeping the newest. Running and parked servers are never dropped."""

    def change(state: dict[str, Any]) -> bool:
        servers = state["servers"]
        # Parked records are not history: they are how Restore finds the server.
        running = [pid_is_running(s.get("pid")) or s.get("status") in (PARKED, RESTORING) for s in servers]
        idle = [i for i, alive in enumerate(running) if not alive]
        excess = len(servers) - max(limit, running.count(True))
        if excess <= 0 or not idle:
            return False
        dropped = set(idle[: min(excess, len(idle))])
        state["servers"] = [s for i, s in enumerate(servers) if i not in dropped]
        return True

    _mutate_state(change)


def _find_server(server_id: str | None = None, mode: str | None = None) -> dict[str, Any] | None:
    servers = list_servers()
    if server_id:
        return next((s for s in servers if s.get("id") == server_id), None)
    if mode:
        # Prefer a live server over a parked record of the same profile.
        matches = [s for s in servers if s.get("mode") == mode]
        return next((s for s in matches if s.get("running")), matches[0] if matches else None)
    return None


def stop_server(server_id: str | None = None, mode: str | None = None, timeout: int = 10) -> dict[str, Any]:
    server = _find_server(server_id, mode)
    if not server:
        return {"success": True, "message": "No tracked server matched the request."}

    if server.get("status") == PARKED:
        _remove_server(server["id"])
        return {"success": True, "message": f"Forgot parked server {server['id']}."}

    raw_pid = server.get("pid")
    if not raw_pid:
        return {"success": True, "message": "Tracked server has no PID to stop."}
    pid = int(raw_pid)
    if not pid_is_running(pid):
        _update_server(server["id"], {"status": "stopped", "running": False, "suspended": False, "stopped_at": _now()})
        return {"success": True, "message": f"Tracked PID {pid} is no longer running."}

    if is_windows():
        cmd = ["taskkill", "/PID", str(pid), "/T", "/F"]
    else:
        if server.get("suspended"):
            # A SIGSTOPped process leaves SIGTERM pending until continued, which
            # would always end in the SIGKILL fallback. Wake it so it can exit cleanly.
            import signal

            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        cmd = ["kill", str(pid)]
    try:
        result = run_hidden(cmd, capture_output=True, text=True, timeout=timeout, check=False)
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
                "suspended": False,
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
        run_hidden(["kill", "-9", str(pid)], capture_output=True, text=True, timeout=timeout, check=False)
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



def _set_process_suspended(pid: int, suspended: bool) -> tuple[bool, str]:
    if not pid_is_running(pid):
        return False, f"PID {pid} is not running."
    if is_windows():
        import ctypes

        PROCESS_SUSPEND_RESUME = 0x0800
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")
        handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, int(pid))
        if not handle:
            return False, f"OpenProcess failed for PID {pid}."
        try:
            fn = ntdll.NtSuspendProcess if suspended else ntdll.NtResumeProcess
            status = int(fn(handle))
            if status != 0:
                return False, f"NT process state change failed with status 0x{status & 0xffffffff:08x}."
        finally:
            kernel32.CloseHandle(handle)
    else:
        import signal

        os.kill(int(pid), signal.SIGSTOP if suspended else signal.SIGCONT)
    return True, (f"Suspended PID {pid}." if suspended else f"Resumed PID {pid}.")


def suspend_server(server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
    server = _find_server(server_id, mode)
    if not server:
        return {"success": False, "error": "No tracked server matched the request."}
    pid = int(server.get("pid") or 0)
    success, message = _set_process_suspended(pid, True)
    if success:
        _update_server(server["id"], {"status": "suspended", "running": True, "suspended": True})
    return {"success": success, "message" if success else "error": message, "server": _find_server(server["id"])}


def resume_server(server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
    server = _find_server(server_id, mode)
    if not server:
        return {"success": False, "error": "No tracked server matched the request."}
    pid = int(server.get("pid") or 0)
    success, message = _set_process_suspended(pid, False)
    if success:
        _update_server(server["id"], {"status": "running", "running": True, "suspended": False})
    return {"success": success, "message" if success else "error": message, "server": _find_server(server["id"])}

def _update_server(server_id: str, patch: dict[str, Any]) -> None:
    def change(state: dict[str, Any]) -> bool:
        servers = state["servers"]
        for idx, server in enumerate(servers):
            if server.get("id") == server_id:
                servers[idx] = {**server, **patch}
                return True
        return False

    _mutate_state(change)


def _upsert_server(server: dict[str, Any]) -> None:
    def change(state: dict[str, Any]) -> None:
        servers = state["servers"]
        for idx, existing in enumerate(servers):
            if existing.get("id") == server.get("id"):
                servers[idx] = server
                return
        servers.append(server)

    _mutate_state(change)


def _remove_server(server_id: str) -> None:
    def change(state: dict[str, Any]) -> bool:
        servers = state["servers"]
        state["servers"] = [s for s in servers if s.get("id") != server_id]
        return len(state["servers"]) != len(servers)

    _mutate_state(change)


# vllm.cpp loads and warms the model before it binds, which takes noticeably
# longer than llama-server (about 53 s cold for a 27B on its reference box).
READY_TIMEOUT_SECONDS = {"llama.cpp": DEFAULT_READY_TIMEOUT_SECONDS, "vllm.cpp": 180}


def _probe_base(host: str, port: int) -> str:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"http://{probe_host}:{int(port)}"


def _probe_status(url: str) -> int | None:
    """HTTP status for ``url``, or None when nothing answered."""
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return None


def wait_until_ready(
    host: str, port: int, pid: int, timeout_seconds: int = DEFAULT_READY_TIMEOUT_SECONDS
) -> bool:
    """Wait for the server to report ready.

    llama-server answers /health with 503 while the model loads and 200 once it
    can serve. Servers without /health (404/405) fall back to /v1/models.
    """
    deadline = time.time() + max(int(timeout_seconds), MIN_READY_TIMEOUT_SECONDS)
    base = _probe_base(host, port)
    path = "/health"
    while time.time() < deadline:
        if not pid_is_running(pid):
            return False
        status = _probe_status(base + path)
        if status is not None and 200 <= status < 300:
            return True
        if path == "/health" and status in {404, 405, 501}:
            path = "/v1/models"
            continue
        time.sleep(1)
    return False


def parse_props(props: dict[str, Any]) -> dict[str, Any]:
    """Capabilities reported by llama-server's GET /props."""
    settings = props.get("default_generation_settings") or {}
    slot_ctx = settings.get("n_ctx")
    total_slots = props.get("total_slots")
    modalities = props.get("modalities") or {}
    template_caps = props.get("chat_template_caps")
    if isinstance(template_caps, dict) and template_caps:
        # Both are needed for a tool round trip (render tools, read calls back).
        tools = bool(template_caps.get("supports_tools") and template_caps.get("supports_tool_calls"))
    else:
        # Builds older than chat_template_caps: the template itself is the only hint.
        tools = "tools" in str(props.get("chat_template") or "")
    caps: dict[str, Any] = {
        "slot_ctx": int(slot_ctx) if isinstance(slot_ctx, int) and slot_ctx > 0 else None,
        "total_slots": int(total_slots) if isinstance(total_slots, int) and total_slots > 0 else None,
        "input_modalities": ["text"] + [name for key, name in _PROPS_MODALITIES if modalities.get(key) is True],
        "tools": tools,
    }
    if props.get("build_info"):
        caps["build_info"] = str(props["build_info"])
    return caps


def probe_capabilities(host: str, port: int, timeout: float = 3) -> dict[str, Any] | None:
    """Read /props from a ready llama-server; None if it has no such endpoint."""
    try:
        with urllib.request.urlopen(f"{_probe_base(host, port)}/props", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return parse_props(payload) if isinstance(payload, dict) else None


def _capability_warnings(caps: dict[str, Any], requested_ctx: Any) -> list[str]:
    slot_ctx = caps.get("slot_ctx")
    try:
        requested = int(requested_ctx)
    except (TypeError, ValueError):
        return []
    if slot_ctx and requested > slot_ctx:
        slots = caps.get("total_slots")
        split = f" across {slots} parallel slots" if slots and slots > 1 else ""
        return [
            f"Requested context {requested} but each request is limited to {slot_ctx} tokens{split}. "
            "Lower --parallel or use a unified KV cache to give one request the full context."
        ]
    return []


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

    params, preset_warnings = layer_sampling_preset(dict(resolved.params), overrides)
    params.setdefault("host", app_config.default_host)
    params.setdefault("port", app_config.default_port)

    runtime = str(params.get("runtime") or "llama.cpp").strip() or "llama.cpp"
    if runtime == "vllm.cpp":
        return _prepare_vllm_cpp(resolved, params, app_config, preset_warnings)
    if runtime != "llama.cpp":
        env = detect_runtime(runtime, root, config=app_config)
        if env is None:
            return {"success": False, "error": f"Unknown runtime: {runtime}"}
        # Report a clear error for runtimes that aren't wired into the launch
        # path instead of silently starting llama.cpp.
        return {
            "success": False,
            "error": (
                f"{env.name} is selected but cannot be launched from here yet — "
                f"only {' and '.join(LAUNCHABLE_RUNTIMES)} can be started. Switch the "
                "profile's runtime to one of those to launch it."
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
    warnings = resolved.warnings + preset_warnings + command.warnings
    return {
        "success": True,
        "runtime": "llama.cpp",
        "profile": resolved.to_dict(),
        "environment": llama.to_dict(),
        "command": command.to_dict(),
        "params": params,
        "request_defaults": request_defaults(params),
        "warnings": warnings,
    }


def _prepare_vllm_cpp(
    resolved: ResolvedProfile, params: dict[str, Any], app_config: AppConfig, preset_warnings: list[str] | None = None
) -> dict[str, Any]:
    env = detect_vllm_cpp(config=app_config)
    if not env.binary_path:
        return {"success": False, "error": "vllm-server (vllm.cpp) was not found.", "environment": env.to_dict()}
    command = build_vllm_cpp_server_args(
        env.binary_path,
        resolved.model["path"],
        params,
        extra_args=app_config.extra_vllm_cpp_args,
    )
    return {
        "success": True,
        "runtime": "vllm.cpp",
        "profile": resolved.to_dict(),
        "environment": env.to_dict(),
        "command": command.to_dict(),
        "params": params,
        "warnings": resolved.warnings + (preset_warnings or []) + command.warnings,
    }


def start_profile(
    mode: str,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    stop_existing: bool = False,
    wait_ready: bool = True,
    ready_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    prepared = prepare_launch_command(mode, project_root, model_dirs, overrides)
    if not prepared.get("success"):
        return prepared
    if ready_timeout_seconds is None:
        ready_timeout_seconds = READY_TIMEOUT_SECONDS.get(prepared.get("runtime") or "llama.cpp", DEFAULT_READY_TIMEOUT_SECONDS)

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
        "runtime": prepared.get("runtime") or "llama.cpp",
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
        # What Release GPU / Restart need to bring this server back as it was.
        "overrides": dict(overrides or {}),
        "ctx_size": params.get("ctx_size"),
        # None falls back to AppConfig.idle_release_seconds (see idle.py).
        "idle_release_seconds": params.get("idle_release_seconds"),
        # Sampling/template defaults baked into the launch flags; requests may override them.
        "request_defaults": prepared.get("request_defaults"),
    }
    _upsert_server(server)
    app_config = AppConfig.load()
    trim_server_history(app_config.server_history_limit)

    if wait_ready:
        ready = wait_until_ready(server["host"], server["port"], proc.pid, ready_timeout_seconds)
        patch: dict[str, Any] = {
            "status": "running" if ready else "startup_timeout",
            "running": pid_is_running(proc.pid),
            "ready_at": _now() if ready else None,
        }
        if ready:
            caps = probe_capabilities(server["host"], server["port"])
            if caps:
                patch["capabilities"] = caps
                extra = _capability_warnings(caps, server.get("ctx_size"))
                if extra:
                    patch["warnings"] = list(server["warnings"]) + extra
        _update_server(server_id, patch)
        if not ready:
            return {
                "success": False,
                "error": "Server process started but did not become ready before timeout.",
                "server": _find_server(server_id),
                "stderr_tail": tail_file(stderr_path),
            }

    return {"success": True, "server": _find_server(server_id), "prepared": prepared}


def release_gpu(server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
    """Stop a server to free its GPU memory, keeping what Restore needs to start it again."""

    server = _find_server(server_id, mode)
    if not server:
        return {"success": False, "error": "No tracked server matched the request."}
    if server.get("status") in (PARKED, RESTORING):
        return {"success": False, "error": "Server is already released.", "server": server}
    stopped = stop_server(server_id=server["id"])
    if not stopped.get("success"):
        return {"success": False, "error": stopped.get("message") or "Could not stop server.", "server": stopped.get("server")}
    parked = {
        **{k: v for k, v in server.items() if k != "running"},
        "pid": None,
        "status": PARKED,
        "suspended": False,
        "parked_at": _now(),
        "restart": {"mode": server.get("mode"), "overrides": dict(server.get("overrides") or {})},
    }
    # Written whole: stop_server may already have pruned the stopped record.
    _upsert_server(parked)
    return {
        "success": True,
        "message": f"Released GPU: stopped {server.get('mode')}. Restore starts it again.",
        "server": _find_server(server["id"]),
    }


def set_idle_release(server_id: str, seconds: int | None) -> dict[str, Any]:
    """Set (or with None, clear back to the config default) a server's idle release window.

    The value is also kept in the server's overrides, so a Restore or Restart
    carries it over.
    """

    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0):
        return {"success": False, "error": "seconds must be a non-negative integer or null."}
    found: dict[str, Any] = {}

    def change(state: dict[str, Any]) -> bool:
        for record in state["servers"]:
            if record.get("id") == server_id:
                overrides = dict(record.get("overrides") or {})
                if seconds is None:
                    overrides.pop("idle_release_seconds", None)
                else:
                    overrides["idle_release_seconds"] = seconds
                record["overrides"] = overrides
                record["idle_release_seconds"] = seconds
                restart = record.get("restart")
                if isinstance(restart, dict):
                    restart["overrides"] = dict(overrides)
                found.update(record)
                return True
        return False

    _mutate_state(change)
    if not found:
        return {"success": False, "error": "No tracked server matched the request."}
    label = "the default" if seconds is None else ("off" if seconds == 0 else f"{seconds}s idle")
    return {"success": True, "message": f"Auto-release set to {label}.", "server": _find_server(server_id)}


def restore_server(
    server_id: str,
    overrides: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Start a parked server again from its saved spec, optionally with changed overrides."""

    claimed: dict[str, Any] = {}

    def claim(state: dict[str, Any]) -> bool:
        for record in state["servers"]:
            if record.get("id") == server_id and record.get("status") == PARKED:
                record["status"] = RESTORING
                claimed.update(record)
                return True
        return False

    # Claiming under the state lock means a double-click can't start two copies.
    _mutate_state(claim)
    if not claimed:
        return {"success": False, "error": "No parked server matched the request."}
    spec = claimed.get("restart") or {}
    merged = {**(spec.get("overrides") or {}), **(overrides or {})}
    result = start_profile(
        str(spec.get("mode") or claimed.get("mode") or ""),
        project_root=project_root,
        model_dirs=model_dirs,
        overrides=merged or None,
    )
    started = result.get("server") or {}
    if result.get("success") or started.get("running"):
        # A server is up (even if it missed the readiness deadline); the parked record is spent.
        _remove_server(server_id)
    else:
        _update_server(server_id, {"status": PARKED})
    return result


def restart_server(
    server_id: str,
    overrides: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    model_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Reload & restart: stop, then start the same profile with optional changed overrides.

    If the new start fails the server stays parked, so Restore can retry.
    """

    server = _find_server(server_id)
    if not server:
        return {"success": False, "error": "No tracked server matched the request."}
    if server.get("status") != PARKED:
        released = release_gpu(server_id=server_id)
        if not released.get("success"):
            return released
    return restore_server(server_id, overrides, project_root=project_root, model_dirs=model_dirs)


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
