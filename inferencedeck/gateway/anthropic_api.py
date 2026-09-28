"""Anthropic Messages API <-> canonical form.

Both directions live here, as in ``openai_api``: clients may send the Messages
API to the gateway (``parse_request``/``render_*``), and the Anthropic engine
sends it upstream (``to_wire``/``from_wire``/``stream_events``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Iterator
from typing import Any

from .ir import (
    FINISH_CONTENT_FILTER,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
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

_STOP_REASONS = {
    FINISH_STOP: "end_turn",
    FINISH_LENGTH: "max_tokens",
    FINISH_TOOL_CALLS: "tool_use",
    FINISH_CONTENT_FILTER: "refusal",
}
_ERROR_TYPES = {
    "invalid_request": "invalid_request_error",
    "authentication": "authentication_error",
    "not_found": "not_found_error",
    "rate_limit": "rate_limit_error",
    "overloaded": "overloaded_error",
    "api_error": "api_error",
}


def _invalid(message: str) -> GatewayError:
    return GatewayError(400, message, "invalid_request")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _image(block: dict[str, Any]) -> ContentPart:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return ContentPart("image", data=str(source.get("data") or ""), media_type=str(source.get("media_type") or ""))
    if source.get("type") == "url":
        return ContentPart("image", url=str(source.get("url") or ""))
    raise _invalid(f"unsupported image source type: {source.get('type')!r}")


def _parse_turn(raw: Any) -> list[Message]:
    """One Anthropic turn; a user turn carrying tool results becomes several messages."""
    if not isinstance(raw, dict) or raw.get("role") not in {"user", "assistant"}:
        raise _invalid("each message needs role user or assistant")
    role = raw["role"]
    content = raw.get("content")
    if isinstance(content, str):
        return [Message(role, [ContentPart("text", text=content)])]
    if not isinstance(content, list):
        raise _invalid("message content must be a string or a list of blocks")
    tool_results: list[Message] = []
    message = Message(role)
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            message.parts.append(ContentPart("text", text=str(block.get("text") or "")))
        elif kind == "image":
            message.parts.append(_image(block))
        elif kind == "tool_use" and role == "assistant":
            message.tool_calls.append(ToolCall(str(block.get("id") or ""), str(block.get("name") or ""),
                                               json.dumps(block.get("input") or {})))
        elif kind == "tool_result" and role == "user":
            text = _text_of(block.get("content"))
            if block.get("is_error"):
                text = f"Error: {text}"
            tool_results.append(Message("tool", [ContentPart("text", text=text)],
                                        tool_call_id=str(block.get("tool_use_id") or "")))
        elif kind in {"thinking", "redacted_thinking"}:
            continue  # prior reasoning is not replayed to other engines
        else:
            raise _invalid(f"unsupported content block type: {kind!r}")
    # Tool results must directly follow the assistant turn that called them.
    return tool_results + ([message] if message.parts or message.tool_calls else [])


def _tool_choice(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    kind = value.get("type")
    if kind == "any":
        return "required"
    if kind == "tool":
        return str(value.get("name") or "")
    return str(kind or "")


def parse_request(body: dict[str, Any]) -> ChatRequest:
    turns = body.get("messages")
    if not isinstance(turns, list) or not turns:
        raise _invalid("messages must be a non-empty list")
    if not isinstance(body.get("max_tokens"), int):
        raise _invalid("max_tokens is required")
    messages: list[Message] = []
    system = _text_of(body.get("system"))
    if system:
        messages.append(Message("system", [ContentPart("text", text=system)]))
    for turn in turns:
        messages.extend(_parse_turn(turn))
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            raise _invalid("each tool needs a name")
        if tool.get("type") not in (None, "custom"):
            raise _invalid(f"server tool {tool['name']!r} is not supported by this gateway")
        tools.append(Tool(str(tool["name"]), str(tool.get("description") or ""),
                          tool.get("input_schema") or {"type": "object", "properties": {}}))
    return ChatRequest(
        model=str(body.get("model") or ""),
        messages=messages,
        sampling=Sampling(
            max_tokens=body["max_tokens"],
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            stop=list(body.get("stop_sequences") or []),
        ),
        tools=tools,
        tool_choice=_tool_choice(body.get("tool_choice")),
        stream=bool(body.get("stream")),
    )


def _message_id(result_id: str) -> str:
    return result_id if result_id.startswith("msg_") else f"msg_{result_id or int(time.time() * 1000)}"


def _tool_input(arguments: str) -> Any:
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def render_result(result: ChatResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.text:
        content.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": _tool_input(call.arguments)})
    return {
        "id": _message_id(result.id),
        "type": "message",
        "role": "assistant",
        "model": result.model,
        "content": content,
        "stop_reason": _STOP_REASONS.get(result.finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens},
    }


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def render_stream(events: Iterable[StreamEvent], model: str) -> Iterator[bytes]:
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": _message_id(""), "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
    }})
    block = -1          # index of the open content block, -1 when none
    block_kind = ""     # "text" or "tool:<canonical index>"
    input_tokens = output_tokens = 0
    finish = FINISH_STOP

    def open_block(kind: str, start: dict[str, Any]) -> Iterator[bytes]:
        nonlocal block, block_kind
        if block >= 0:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block})
        block += 1
        block_kind = kind
        yield _sse("content_block_start", {"type": "content_block_start", "index": block, "content_block": start})

    try:
        for event in events:
            if event.kind == "text" and event.text:
                if block_kind != "text":
                    yield from open_block("text", {"type": "text", "text": ""})
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": block,
                                                   "delta": {"type": "text_delta", "text": event.text}})
            elif event.kind == "tool_call":
                kind = f"tool:{event.index}"
                if block_kind != kind:
                    yield from open_block(kind, {"type": "tool_use", "id": event.tool_id,
                                                 "name": event.tool_name, "input": {}})
                if event.arguments:
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": block,
                                                       "delta": {"type": "input_json_delta",
                                                                 "partial_json": event.arguments}})
            elif event.kind == "usage" and event.usage:
                input_tokens, output_tokens = event.usage.input_tokens, event.usage.output_tokens
            elif event.kind == "finish":
                finish = event.finish_reason
    except GatewayError as exc:
        yield _sse("error", render_error(exc))
        return
    if block >= 0:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _STOP_REASONS.get(finish, "end_turn"), "stop_sequence": None},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def render_error(error: GatewayError) -> dict[str, Any]:
    return {"type": "error", "error": {"type": _ERROR_TYPES.get(error.kind, "api_error"), "message": error.message}}


# ---- canonical -> Anthropic wire (for engines) -----------------------------

# The API requires max_tokens; used when the client did not set one. Streaming
# gets more room because it is not bound by an HTTP read timeout.
DEFAULT_MAX_TOKENS = 16000
DEFAULT_STREAM_MAX_TOKENS = 64000

_FINISH_REASONS = {
    "end_turn": FINISH_STOP,
    "stop_sequence": FINISH_STOP,
    "pause_turn": FINISH_STOP,
    "max_tokens": FINISH_LENGTH,
    "model_context_window_exceeded": FINISH_LENGTH,
    "tool_use": FINISH_TOOL_CALLS,
    "refusal": FINISH_CONTENT_FILTER,
}
_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.S)


def _tool_id(raw: str, index: int) -> str:
    # tool_use ids must match ^[a-zA-Z0-9_-]+$; ids from other engines may not.
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw) or f"toolu_gateway_{index}"


def _wire_image(part: ContentPart) -> dict[str, Any]:
    if part.data:
        source = {"type": "base64", "media_type": part.media_type or "image/png", "data": part.data}
    else:
        match = _DATA_URL.match(part.url)
        if match:
            source = {"type": "base64", "media_type": match.group(1), "data": match.group(2)}
        else:
            source = {"type": "url", "url": part.url}
    return {"type": "image", "source": source}


def _wire_blocks(message: Message, ids: dict[str, str]) -> list[dict[str, Any]]:
    if message.role == "tool":
        # Tool results travel as a user turn; tool_use_id must match the call.
        return [{"type": "tool_result", "tool_use_id": ids.get(message.tool_call_id, _tool_id(message.tool_call_id, 0)),
                 "content": message.text}]
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if part.type == "image":
            blocks.append(_wire_image(part))
        elif part.text:  # the API rejects empty text blocks
            blocks.append({"type": "text", "text": part.text})
    for call in message.tool_calls:
        blocks.append({"type": "tool_use", "id": ids[call.id], "name": call.name, "input": _tool_input(call.arguments)})
    return blocks


def _wire_tool_choice(choice: str) -> dict[str, Any] | None:
    if not choice:
        return None
    if choice in {"auto", "none"}:
        return {"type": choice}
    if choice == "required":
        return {"type": "any"}
    return {"type": "tool", "name": choice}


def to_wire(request: ChatRequest, model: str) -> dict[str, Any]:
    ids: dict[str, str] = {}
    for message in request.messages:
        for call in message.tool_calls:
            ids.setdefault(call.id, _tool_id(call.id, len(ids)))
    system = "\n\n".join(m.text for m in request.messages if m.role == "system" and m.text)
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            continue  # folded into the top-level system prompt
        role = "assistant" if message.role == "assistant" else "user"
        blocks = _wire_blocks(message, ids)
        if not blocks:
            continue
        if messages and messages[-1]["role"] == role:
            # One turn per role: parallel tool results, and text sent after
            # them, belong in a single user message.
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    sampling = request.sampling
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": sampling.max_tokens or (DEFAULT_STREAM_MAX_TOKENS if request.stream else DEFAULT_MAX_TOKENS),
        "stream": request.stream,
    }
    if system:
        body["system"] = system
    # Only fields the Messages API accepts: it rejects unknown ones, so min_p,
    # penalties, seed and engine extras are not forwarded.
    for name in ("temperature", "top_p", "top_k"):
        value = getattr(sampling, name)
        if value is not None:
            body[name] = value
    if sampling.stop:
        body["stop_sequences"] = sampling.stop
    if request.tools:
        body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                         for t in request.tools]
        choice = _wire_tool_choice(request.tool_choice)
        if choice is not None:
            body["tool_choice"] = choice
    fmt = request.response_format if isinstance(request.response_format, dict) else {}
    schema = (fmt.get("json_schema") or {}).get("schema") if fmt.get("type") == "json_schema" else None
    if isinstance(schema, dict):
        body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    return body


def _wire_usage(raw: Any) -> Usage:
    raw = raw if isinstance(raw, dict) else {}
    # Cached prompt tokens are reported separately; the client wants the total.
    prompt = sum(int(raw.get(key) or 0) for key in
                 ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return Usage(prompt, int(raw.get("output_tokens") or 0))


def from_wire(payload: dict[str, Any], model: str) -> ChatResult:
    text: list[str] = []
    calls: list[ToolCall] = []
    for block in payload.get("content") or []:
        if block.get("type") == "text":
            text.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            calls.append(ToolCall(str(block.get("id") or ""), str(block.get("name") or ""),
                                  json.dumps(block.get("input") or {})))
        # thinking / redacted_thinking and server-tool blocks have no canonical slot.
    return ChatResult(
        id=str(payload.get("id") or ""),
        model=str(payload.get("model") or model),
        text="".join(text),
        tool_calls=calls,
        finish_reason=_FINISH_REASONS.get(str(payload.get("stop_reason") or ""), FINISH_STOP),
        usage=_wire_usage(payload.get("usage")),
    )


def stream_events(lines: Iterable[bytes]) -> Iterator[StreamEvent]:
    """Canonical events from a Messages API SSE stream."""
    tool_index: dict[int, int] = {}  # content block index -> canonical tool-call index
    usage = Usage()
    finish = FINISH_STOP
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue  # event: names repeat the data's "type"
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "message_start":
            usage = _wire_usage((event.get("message") or {}).get("usage"))
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = tool_index[int(event.get("index") or 0)] = len(tool_index)
                yield StreamEvent("tool_call", index=index, tool_id=str(block.get("id") or ""),
                                  tool_name=str(block.get("name") or ""))
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield StreamEvent("text", text=str(delta["text"]))
            elif delta.get("type") == "input_json_delta" and delta.get("partial_json"):
                block_index = int(event.get("index") or 0)
                if block_index in tool_index:
                    yield StreamEvent("tool_call", index=tool_index[block_index], arguments=str(delta["partial_json"]))
            # thinking_delta / signature_delta have no canonical slot.
        elif kind == "message_delta":
            reason = (event.get("delta") or {}).get("stop_reason")
            if reason:
                finish = _FINISH_REASONS.get(str(reason), FINISH_STOP)
            final = event.get("usage") or {}
            if final.get("output_tokens") is not None:
                usage.output_tokens = int(final["output_tokens"])
        elif kind == "error":
            error = event.get("error") or {}
            raise GatewayError(502, f"upstream: {error.get('message') or 'stream error'}",
                               "overloaded" if error.get("type") == "overloaded_error" else "api_error")
        elif kind == "message_stop":
            break
    yield StreamEvent("usage", usage=usage)
    yield StreamEvent("finish", finish_reason=finish)
