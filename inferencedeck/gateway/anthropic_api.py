"""Anthropic Messages API (client side) <-> canonical form."""

from __future__ import annotations

import json
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
