"""OpenAI Chat Completions API <-> canonical form.

The same wire format is used in both directions: clients send it to the
gateway (``parse_request``/``render_*``) and OpenAI-compatible engines receive
it (``to_wire``/``from_wire``/``stream_events``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from typing import Any

from .ir import (
    FINISH_STOP,
    ChatRequest,
    ChatResult,
    ContentPart,
    GatewayError,
    Message,
    Sampling,
    StreamEvent,
    Tool,
    ToolCall,
    Usage,
)

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed")
# Fields parse_request handles itself; anything else goes to ChatRequest.extra.
_KNOWN_FIELDS = {
    "model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "stop",
    "tools", "tool_choice", "response_format", "n", "user", "functions", "function_call",
    *_SAMPLING_FIELDS,
}
_ERROR_TYPES = {
    "invalid_request": "invalid_request_error",
    "authentication": "authentication_error",
    "not_found": "not_found_error",
    "rate_limit": "rate_limit_error",
    "overloaded": "server_error",
    "api_error": "server_error",
}


def _invalid(message: str) -> GatewayError:
    return GatewayError(400, message, "invalid_request")


def _parts(content: Any) -> list[ContentPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [ContentPart("text", text=content)]
    if not isinstance(content, list):
        raise _invalid("message content must be a string or a list of parts")
    parts = []
    for item in content:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            parts.append(ContentPart("text", text=str(item.get("text") or "")))
        elif kind == "image_url":
            image = item.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            parts.append(ContentPart("image", url=str(url or "")))
        else:
            raise _invalid(f"unsupported content part type: {kind!r}")
    return parts


def _parse_message(raw: Any) -> Message:
    if not isinstance(raw, dict):
        raise _invalid("each message must be an object")
    role = str(raw.get("role") or "")
    if role == "developer":
        role = "system"
    if role not in {"system", "user", "assistant", "tool"}:
        raise _invalid(f"unsupported message role: {role!r}")
    calls = []
    for call in raw.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append(ToolCall(str(call.get("id") or ""), str(function.get("name") or ""), str(function.get("arguments") or "{}")))
    return Message(role, _parts(raw.get("content")), calls, str(raw.get("tool_call_id") or ""))


def _tool_choice(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str((value.get("function") or {}).get("name") or "")
    raise _invalid("tool_choice must be a string or an object")


def parse_request(body: dict[str, Any]) -> ChatRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _invalid("messages must be a non-empty list")
    if body.get("n") not in (None, 1):
        raise _invalid("only n=1 is supported")
    stop = body.get("stop")
    sampling = Sampling(
        max_tokens=body.get("max_completion_tokens", body.get("max_tokens")),
        stop=[stop] if isinstance(stop, str) else list(stop or []),
        **{name: body[name] for name in _SAMPLING_FIELDS if body.get(name) is not None},
    )
    tools = []
    for tool in body.get("tools") or []:
        function = (tool.get("function") or {}) if isinstance(tool, dict) else {}
        if not function.get("name"):
            raise _invalid("each tool needs function.name")
        tools.append(Tool(str(function["name"]), str(function.get("description") or ""),
                          function.get("parameters") or {"type": "object", "properties": {}}))
    return ChatRequest(
        model=str(body.get("model") or ""),
        messages=[_parse_message(message) for message in messages],
        sampling=sampling,
        tools=tools,
        tool_choice=_tool_choice(body.get("tool_choice")),
        stream=bool(body.get("stream")),
        response_format=body.get("response_format"),
        extra={key: value for key, value in body.items() if key not in _KNOWN_FIELDS},
    )


def wants_usage_in_stream(body: dict[str, Any]) -> bool:
    options = body.get("stream_options")
    return isinstance(options, dict) and bool(options.get("include_usage"))


# ---- canonical -> OpenAI wire (for engines) -------------------------------


def _wire_content(message: Message) -> Any:
    if all(part.type == "text" for part in message.parts):
        return message.text if message.parts or message.role != "assistant" else None
    wire = []
    for part in message.parts:
        if part.type == "text":
            wire.append({"type": "text", "text": part.text})
        else:
            url = part.url or f"data:{part.media_type or 'image/png'};base64,{part.data}"
            wire.append({"type": "image_url", "image_url": {"url": url}})
    return wire


def _wire_message(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": _wire_content(message)}
    if message.tool_calls:
        wire["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    if message.role == "tool":
        wire["tool_call_id"] = message.tool_call_id
    return wire


def to_wire(request: ChatRequest, model: str) -> dict[str, Any]:
    body: dict[str, Any] = dict(request.extra)
    body.update({"model": model, "messages": [_wire_message(m) for m in request.messages], "stream": request.stream})
    sampling = request.sampling
    if sampling.max_tokens is not None:
        body["max_tokens"] = sampling.max_tokens
    if sampling.stop:
        body["stop"] = sampling.stop
    for name in _SAMPLING_FIELDS:
        value = getattr(sampling, name)
        if value is not None:
            body[name] = value
    if request.tools:
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in request.tools
        ]
    if request.tool_choice in {"auto", "none", "required"}:
        body["tool_choice"] = request.tool_choice
    elif request.tool_choice:
        body["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}
    if request.response_format is not None:
        body["response_format"] = request.response_format
    if request.stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _usage(raw: Any) -> Usage:
    raw = raw if isinstance(raw, dict) else {}
    return Usage(int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0))


def from_wire(payload: dict[str, Any], model: str) -> ChatResult:
    choices = payload.get("choices") or []
    if not choices:
        raise GatewayError(502, "upstream returned no choices")
    choice = choices[0] or {}
    message = choice.get("message") or {}
    calls = [
        ToolCall(str(c.get("id") or ""), str((c.get("function") or {}).get("name") or ""),
                 str((c.get("function") or {}).get("arguments") or "{}"))
        for c in message.get("tool_calls") or []
    ]
    return ChatResult(
        id=str(payload.get("id") or ""),
        model=str(payload.get("model") or model),
        text=str(message.get("content") or ""),
        tool_calls=calls,
        finish_reason=str(choice.get("finish_reason") or FINISH_STOP),
        usage=_usage(payload.get("usage")),
    )


def stream_events(lines: Iterable[bytes]) -> Iterator[StreamEvent]:
    """Canonical events from an OpenAI-format SSE byte stream."""
    finish = ""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk.get("error"), dict):
            raise GatewayError(502, str(chunk["error"].get("message") or "upstream stream error"))
        if chunk.get("usage"):
            yield StreamEvent("usage", usage=_usage(chunk["usage"]))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield StreamEvent("text", text=str(delta["content"]))
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                yield StreamEvent(
                    "tool_call",
                    index=int(call.get("index") or 0),
                    tool_id=str(call.get("id") or ""),
                    tool_name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or ""),
                )
            if choice.get("finish_reason"):
                finish = str(choice["finish_reason"])
    yield StreamEvent("finish", finish_reason=finish or FINISH_STOP)


# ---- canonical -> OpenAI responses (for clients) ---------------------------


def render_result(result: ChatResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text or (None if result.tool_calls else "")}
    if result.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in result.tool_calls
        ]
    return {
        "id": result.id or f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason}],
        "usage": {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
        },
    }


def _sse(payload: Any) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def render_stream(events: Iterable[StreamEvent], model: str, include_usage: bool) -> Iterator[bytes]:
    stream_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
        return _sse({
            "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    yield chunk({"role": "assistant", "content": ""})
    usage = None
    try:
        for event in events:
            if event.kind == "text":
                yield chunk({"content": event.text})
            elif event.kind == "tool_call":
                call: dict[str, Any] = {"index": event.index, "function": {"arguments": event.arguments}}
                if event.tool_id:
                    call["id"] = event.tool_id
                    call["type"] = "function"
                if event.tool_name:
                    call["function"]["name"] = event.tool_name
                yield chunk({"tool_calls": [call]})
            elif event.kind == "usage":
                usage = event.usage
            elif event.kind == "finish":
                yield chunk({}, event.finish_reason)
    except GatewayError as exc:
        # Headers are already sent; report the failure in-band.
        yield _sse(render_error(exc))
    except Exception as exc:
        # E.g. an upstream chunk of an unexpected shape: still an in-band error,
        # never an exception out of a response that has already started.
        yield _sse(render_error(GatewayError(502, f"upstream stream failed: {exc}")))
    if include_usage and usage is not None:
        yield _sse({
            "id": stream_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
            "usage": {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens,
                      "total_tokens": usage.input_tokens + usage.output_tokens},
        })
    yield b"data: [DONE]\n\n"


def render_error(error: GatewayError) -> dict[str, Any]:
    return {"error": {"message": error.message, "type": _ERROR_TYPES.get(error.kind, "server_error"), "code": None}}
