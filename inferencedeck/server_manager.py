from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import detect_llama_cpp, detect_runtime
from .config import AppConfig
from .llama_args import LaunchCommand, build_llama_server_args
from .paths import cache_dir, find_project_root, is_windows
from .profile_resolver import ResolvedProfile, resolve_profiles
from .proc import run as run_hidden


STATE_FILENAME = "servers.json"
# A parked server was stopped to free its GPU memory; its record keeps the
# restart spec (mode + overrides) so Restore can bring it back.
PARKED = "parked"
RESTORING = "restoring"
CONTEXT_PRESETS = (8192, 16384, 32768, 65536, 131072)
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


# Serializes start_profile's check-then-launch. Separate from the state lock,
# which is only held briefly, so status polls never wait on a server start.
_LAUNCH_LOCK = threading.Lock()


@contextmanager
def launch_lock() -> Iterator[None]:
    with _LAUNCH_LOCK:
        with open(state_path().with_name("launch.lock"), "a+b") as fh:
            _lock_file(fh)
            try:
                yield
            finally:
                _unlock_file(fh)


WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


def _same_listener(server: dict[str, Any], host: str, port: int) -> bool:
    """Whether ``server`` listens where a new server on host:port would."""
    if int(server.get("port") or 0) != port:
        return False
    other = str(server.get("host") or "")
    return other == host or other in WILDCARD_HOSTS or host in WILDCARD_HOSTS


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


def _windows_creation_time(pid: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        created = times[0]
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def process_identity(pid: int | None) -> str | None:
    """A token that tells this process apart from a later one given the same PID.

    It is the process start time (plus the boot id on Linux, whose start time
    counts from boot). None when it can't be read; callers then fall back to
    the PID alone.
    """
    if not pid:
        return None
    pid = int(pid)
    try:
        if is_windows():
            return _windows_creation_time(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as f:
                # Field 22 (starttime); fields are counted after the ")" that ends comm.
                start = f.read().rpartition(")")[2].split()[19]
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
                return f"{f.read().strip()}:{start}"
        except FileNotFoundError:
            pass
        result = run_hidden(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() or None
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _is_same_process(pid: int | None, identity: str | None) -> bool:
    """False only when ``pid`` now provably belongs to a different process than ``identity``."""
    if not identity:
        return True  # records written before identities were kept
    current = process_identity(pid)
    return current is None or current == identity


def server_alive(server: dict[str, Any]) -> bool:
    """The record's process is still running, and it is the process we started."""
    pid = server.get("pid")
    return pid_is_running(pid) and _is_same_process(pid, server.get("pid_identity"))


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
        item["running"] = server_alive(item)
        servers.append(item)
    return servers


def prune_stale_servers() -> None:
    """Remove entries for PIDs that are no longer running."""

    def change(state: dict[str, Any]) -> bool:
        servers = state["servers"]
        kept = [s for s in servers if server_alive(s) or not s.get("pid")]
        state["servers"] = kept
        return len(kept) != len(servers)

    _mutate_state(change)


def trim_server_history(limit: int = 5) -> None:
    """Cap non-running records at ``limit``, keeping the newest. Running and parked servers are never dropped."""

    def change(state: dict[str, Any]) -> bool:
        servers = state["servers"]
        # Parked records are not history: they are how Restore finds the server.
        running = [server_alive(s) or s.get("status") in (PARKED, RESTORING) for s in servers]
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
    if not server_alive(server):
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



def _set_process_suspended(pid: int, suspended: bool, identity: str | None = None) -> tuple[bool, str]:
    if not pid_is_running(pid) or not _is_same_process(pid, identity):
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
    success, message = _set_process_suspended(pid, True, server.get("pid_identity"))
    if success:
        _update_server(server["id"], {"status": "suspended", "running": True, "suspended": True})
    return {"success": success, "message" if success else "error": message, "server": _find_server(server["id"])}


def resume_server(server_id: str | None = None, mode: str | None = None) -> dict[str, Any]:
    server = _find_server(server_id, mode)
    if not server:
        return {"success": False, "error": "No tracked server matched the request."}
    pid = int(server.get("pid") or 0)
    success, message = _set_process_suspended(pid, False, server.get("pid_identity"))
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

    # Held from the "already running?" check until the new server is recorded,
    # so two Start requests (double-click, tray + browser, CLI + daemon) can't
    # both pass the check and launch two servers on one port.
    with launch_lock():
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

        params = prepared["params"]
        host, port = str(params.get("host", "127.0.0.1")), int(params.get("port", 8080))
        clash = next((s for s in list_servers() if s.get("running") and _same_listener(s, host, port)), None)
        if clash:
            return {
                "success": False,
                "error": f"Port {port} is already used by tracked server '{clash.get('mode') or clash.get('id')}'.",
                "server": clash,
            }

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

        server_id = f"{mode}-{proc.pid}"
        server = {
            "id": server_id,
            "mode": mode,
            "pid": proc.pid,
            # Start time of this process, so a later process reusing the PID is
            # never mistaken for it (and never stopped or paused by us).
            "pid_identity": process_identity(proc.pid),
            "status": "starting",
            "running": True,
            "host": host,
            "port": port,
            "model_path": prepared["profile"]["model"]["path"] if prepared["profile"].get("model") else None,
            "command_line": command.command_line,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "started_at": _now(),
            "warnings": prepared.get("warnings", []),
            # What Release GPU / Restart need to bring this server back as it was.
            "overrides": dict(overrides or {}),
            "ctx_size": params.get("ctx_size"),
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
        "pid_identity": None,
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
