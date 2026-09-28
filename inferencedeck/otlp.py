"""Optional OpenTelemetry export over OTLP/HTTP with JSON encoding.

Off unless an endpoint is configured. It sends to an OpenTelemetry Collector (or
any OTLP/HTTP receiver, usually port 4318) and needs no extra dependencies:
the JSON encoding of OTLP is plain ``urllib``.

- Metrics: the same readings ``/metrics`` serves, sent every
  ``otlp_export_seconds``. Gauges stay gauges; counters become cumulative sums.
- Logs: every lifecycle event, as a log record.
- Traces: a span per server startup (launch until ready, or until it gave up)
  and per benchmark run, timed from what the event recorded.

Configuration follows the standard OpenTelemetry environment variables where
they apply: ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``
(keep credentials here rather than in config.json), ``OTEL_SERVICE_NAME`` and
``OTEL_RESOURCE_ATTRIBUTES``. Only the http/json protocol is spoken.

Like the rest of telemetry, export never gates the lifecycle: a collector
that is down only costs dropped data, counted in ``status()``.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from typing import Any, Callable

from . import __version__, telemetry

DEFAULT_EXPORT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 5.0
MAX_QUEUED_RECORDS = 2000
SCOPE = {"name": "inferencedeck", "version": __version__}

# Severity numbers from the OTLP logs data model.
SEVERITY_INFO, SEVERITY_WARN = 9, 13
FAILED_EVENTS = {"server.start_failed", "server.stop_failed"}
SPAN_KIND_INTERNAL, STATUS_OK, STATUS_ERROR = 1, 1, 2
CUMULATIVE = 2

# Prometheus-style name suffix -> UCUM unit.
UNIT_SUFFIXES = (
    ("_bytes", "By"),
    ("_seconds_total", "s"),
    ("_seconds", "s"),
    ("_percent", "%"),
    ("_celsius", "Cel"),
    ("_watts", "W"),
    ("_mhz", "MHz"),
    ("_tokens_per_second", "{token}/s"),
    ("_tokens_total", "{token}"),
    ("_ratio", "1"),
)


def parse_key_values(text: str) -> dict[str, str]:
    """``k1=v1,k2=v2`` with percent-encoded values, as the OTEL_* variables use."""
    pairs: dict[str, str] = {}
    for item in (text or "").split(","):
        key, sep, value = item.partition("=")
        if sep and key.strip():
            pairs[urllib.parse.unquote(key.strip())] = urllib.parse.unquote(value.strip())
    return pairs


def resolve_endpoint(configured: str = "") -> str:
    """The OTLP base URL, from config or OTEL_EXPORTER_OTLP_ENDPOINT; '' when export is off."""
    endpoint = (configured or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip().rstrip("/")
    if not endpoint:
        return ""
    if urllib.parse.urlparse(endpoint).scheme not in ("http", "https"):
        raise ValueError(f"OTLP endpoint must be an http:// or https:// URL, not {endpoint!r}")
    return endpoint


def _nanos(t: float) -> str:
    # OTLP JSON carries 64-bit integers as decimal strings.
    return str(int(t * 1_000_000_000))


def _any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": str(k), "value": _any_value(v)} for k, v in values.items() if v is not None]


def _unit(name: str) -> str:
    return next((unit for suffix, unit in UNIT_SUFFIXES if name.endswith(suffix)), "")


def metrics_payload(samples: list[telemetry.Sample], resource: list[dict[str, Any]], now: float) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for sample in samples:
        metric = metrics.get(sample.name)
        if metric is None:
            metric = {"name": sample.name, "description": sample.help, "unit": _unit(sample.name)}
            if sample.kind == "counter":
                metric["sum"] = {"aggregationTemporality": CUMULATIVE, "isMonotonic": True, "dataPoints": []}
            else:
                metric["gauge"] = {"dataPoints": []}
            metrics[sample.name] = metric
        point: dict[str, Any] = {"attributes": _attributes(sample.labels), "timeUnixNano": _nanos(now), "asDouble": sample.value}
        if sample.kind == "counter":
            point["startTimeUnixNano"] = _nanos(sample.start if sample.start is not None else now)
        (metric.get("sum") or metric["gauge"])["dataPoints"].append(point)
    return {"resourceMetrics": [{"resource": {"attributes": resource}, "scopeMetrics": [{"scope": SCOPE, "metrics": list(metrics.values())}]}]}


def _event_time(event: dict[str, Any]) -> float:
    return telemetry._parse_iso(event.get("timestamp")) or time.time()


def _event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    names = {"server_id": "inferencedeck.server.id", "profile": "inferencedeck.profile", "runtime": "inferencedeck.runtime", "pid": "process.pid"}
    return {names.get(k, f"inferencedeck.{k}"): v for k, v in event.items() if k not in ("type", "timestamp")}


def log_record(event: dict[str, Any]) -> dict[str, Any]:
    t = _nanos(_event_time(event))
    failed = event.get("type") in FAILED_EVENTS
    summary = " ".join(str(part) for part in (event.get("type"), event.get("profile"), event.get("reason")) if part)
    return {
        "timeUnixNano": t,
        "observedTimeUnixNano": t,
        "severityNumber": SEVERITY_WARN if failed else SEVERITY_INFO,
        "severityText": "WARN" if failed else "INFO",
        "body": {"stringValue": summary},
        "attributes": _attributes({"event.name": event.get("type"), **_event_attributes(event)}),
    }


def span_for(event: dict[str, Any]) -> dict[str, Any] | None:
    """A span for events that close a timed operation; None for the rest."""
    kind = event.get("type")
    duration_key = {"server.ready": "startup_seconds", "server.start_failed": "waited_seconds", "benchmark.completed": "elapsed_seconds"}.get(kind)
    duration = event.get(duration_key) if duration_key else None
    if not isinstance(duration, (int, float)) or duration < 0:
        return None
    end = _event_time(event)
    failed = kind == "server.start_failed"
    return {
        "traceId": secrets.token_hex(16),
        "spanId": secrets.token_hex(8),
        "name": "inferencedeck.benchmark" if kind == "benchmark.completed" else "inferencedeck.server.startup",
        "kind": SPAN_KIND_INTERNAL,
        "startTimeUnixNano": _nanos(end - duration),
        "endTimeUnixNano": _nanos(end),
        "attributes": _attributes(_event_attributes(event)),
        "status": {"code": STATUS_ERROR, "message": str(event.get("reason") or "failed")} if failed else {"code": STATUS_OK},
    }


def default_resource() -> dict[str, Any]:
    attributes = {"service.name": os.environ.get("OTEL_SERVICE_NAME") or "inferencedeck", "service.version": __version__}
    try:
        attributes["host.name"] = socket.gethostname()
    except OSError:
        pass
    attributes.update(parse_key_values(os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")))
    if os.environ.get("OTEL_SERVICE_NAME"):
        attributes["service.name"] = os.environ["OTEL_SERVICE_NAME"]  # the dedicated variable wins
    return attributes


Post = Callable[[str, bytes, dict[str, str]], None]


def _urllib_post(url: str, body: bytes, headers: dict[str, str]) -> None:
    request = urllib.request.Request(url, data=body, method="POST", headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.read()


class OtlpExporter:
    def __init__(
        self,
        endpoint: str,
        interval: float = DEFAULT_EXPORT_SECONDS,
        headers: dict[str, str] | None = None,
        resource: dict[str, Any] | None = None,
        collect: Callable[[], dict[str, Any]] | None = None,
        post: Post | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.interval = max(1.0, float(interval))
        self.headers = dict(headers if headers is not None else parse_key_values(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")))
        self._resource = _attributes(resource if resource is not None else default_resource())
        self._collect = collect or telemetry.snapshot
        self._post = post or _urllib_post
        self._lock = threading.Lock()
        self._logs: deque[dict[str, Any]] = deque()
        self._spans: deque[dict[str, Any]] = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {"last_export_at": None, "last_error": None, "exported": {"metrics": 0, "logs": 0, "spans": 0}, "dropped": 0}

    # --- queueing -----------------------------------------------------------

    def _queue(self, target: deque, item: dict[str, Any]) -> None:
        target.append(item)
        while len(self._logs) + len(self._spans) > MAX_QUEUED_RECORDS:
            (self._logs or self._spans).popleft()
            self._status["dropped"] += 1

    def record_event(self, event: dict[str, Any]) -> None:
        try:
            log, span = log_record(event), span_for(event)
        except Exception:
            return
        with self._lock:
            self._queue(self._logs, log)
            if span:
                self._queue(self._spans, span)

    # --- sending ------------------------------------------------------------

    def _send(self, path: str, payload: dict[str, Any]) -> None:
        self._post(f"{self.endpoint}{path}", json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.headers)

    def export_once(self) -> bool:
        """Send one round; True when everything went through."""
        now = time.time()
        errors: list[str] = []
        try:
            samples = telemetry.metric_samples(self._collect())
            if samples:
                self._send("/v1/metrics", metrics_payload(samples, self._resource, now))
                self._status["exported"]["metrics"] += len(samples)
        except Exception as exc:
            # Metrics are a point-in-time reading: the next round supersedes a lost one.
            errors.append(f"metrics: {exc}")
        for path, key, queue, wrap in (
            ("/v1/logs", "logs", self._logs, lambda items: {"resourceLogs": [{"resource": {"attributes": self._resource}, "scopeLogs": [{"scope": SCOPE, "logRecords": items}]}]}),
            ("/v1/traces", "spans", self._spans, lambda items: {"resourceSpans": [{"resource": {"attributes": self._resource}, "scopeSpans": [{"scope": SCOPE, "spans": items}]}]}),
        ):
            with self._lock:
                batch = list(queue)
                queue.clear()
            if not batch:
                continue
            try:
                self._send(path, wrap(batch))
                self._status["exported"][key] += len(batch)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                with self._lock:  # keep events for the next round, oldest first
                    for item in reversed(batch):
                        queue.appendleft(item)
                    while len(self._logs) + len(self._spans) > MAX_QUEUED_RECORDS:
                        queue.popleft()
                        self._status["dropped"] += 1
        with self._lock:
            if errors:
                self._status["last_error"] = "; ".join(errors)
            else:
                self._status["last_export_at"] = telemetry._now_iso()
                self._status["last_error"] = None
        return not errors

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "endpoint": self.endpoint,
                "protocol": "http/json",
                "interval_seconds": self.interval,
                "queued": len(self._logs) + len(self._spans),
                **json.loads(json.dumps(self._status)),
            }

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        telemetry.REGISTRY.add_listener(self.record_event)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="inferencedeck-otlp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        telemetry.REGISTRY.remove_listener(self.record_event)
        if self._thread:
            self._thread.join(timeout=REQUEST_TIMEOUT_SECONDS * 3 + 1)
        self.export_once()  # flush what's queued

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.export_once()
            except Exception:
                pass


_exporter: OtlpExporter | None = None
_exporter_lock = threading.Lock()


def start_exporter(endpoint: str = "", interval: float = DEFAULT_EXPORT_SECONDS) -> OtlpExporter | None:
    """Start the process-wide exporter once. None when no endpoint is configured."""
    global _exporter
    resolved = resolve_endpoint(endpoint)
    if not resolved:
        return None
    with _exporter_lock:
        if _exporter is None:
            _exporter = OtlpExporter(resolved, interval=interval)
            _exporter.start()
        return _exporter


def status() -> dict[str, Any]:
    exporter = _exporter
    return exporter.status() if exporter else {"enabled": False}
