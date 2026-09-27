"""Canonical form <-> Ollama's native chat API (``POST /api/chat``).

Used by the Ollama engine. Differences from the OpenAI format that matter here:
sampling lives under ``options`` (``max_tokens`` is ``num_predict``), images
are bare base64 strings on the message, tool-call arguments are JSON objects
rather than text and carry no ids, structured output is ``format``, and
streaming is newline-delimited JSON rather than SSE.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from .ir import (
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    ChatRequest,
    ChatResult,
    GatewayError,
    Message,
    StreamEvent,
    ToolCall,
    Usage,
)

# Canonical sampling field -> Ollama option name.
_OPTIONS = {
    "max_tokens": "num_predict",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}
# Pass-through fields that belong at the top level of an Ollama request;
# every other extra (repeat_penalty, num_ctx, mirostat, ...) is a model option.
_TOP_LEVEL_EXTRAS = {"keep_alive", "think"}
# OpenAI-only request fields with no Ollama equivalent; dropped, not sent as options.
_OPENAI_ONLY = {"parallel_tool_calls", "logprobs", "top_logprobs", "logit_bias", "store", "metadata",
                "service_tier", "reasoning_effort", "modalities", "prediction", "audio"}


def _image_data(url: str, data: str) -> str:
    if data:
        return data
    if url.startswith("data:") and ";base64," in url:
        return url.split(";base64,", 1)[1]
    raise GatewayError(400, "Ollama accepts inline (base64) images only, not image URLs", "invalid_request")


def _arguments(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _wire_message(message: Message, tool_names: dict[str, str]) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.text}
    images = [_image_data(p.url, p.data) for p in message.parts if p.type == "image"]
    if images:
        wire["images"] = images
    if message.tool_calls:
        wire["tool_calls"] = [{"function": {"name": c.name, "arguments": _arguments(c.arguments)}}
                              for c in message.tool_calls]
    if message.role == "tool" and message.tool_call_id in tool_names:
        # Ollama matches results to calls by tool name, not id.
        wire["tool_name"] = tool_names[message.tool_call_id]
    return wire


def _format(response_format: dict[str, Any] | None) -> Any:
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return "json"
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        return schema if isinstance(schema, dict) else "json"
    return None


def to_wire(request: ChatRequest, model: str) -> dict[str, Any]:
    tool_names = {call.id: call.name for m in request.messages for call in m.tool_calls if call.id}
    options: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for key, value in request.extra.items():
        if key in _OPENAI_ONLY:
            continue
        (body if key in _TOP_LEVEL_EXTRAS else options)[key] = value
    for field, option in _OPTIONS.items():
        value = getattr(request.sampling, field)
        if value is not None:
            options[option] = value
    if request.sampling.stop:
        options["stop"] = request.sampling.stop
    body.update({
        "model": model,
        "messages": [_wire_message(m, tool_names) for m in request.messages],
        # Ollama streams unless told otherwise.
        "stream": request.stream,
    })
    if options:
        body["options"] = options
    # Ollama has no tool_choice; "none" is honoured by not offering tools,
    # and a forced or required choice is left to the model.
    if request.tools and request.tool_choice != "none":
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in request.tools
        ]
    fmt = _format(request.response_format)
    if fmt is not None:
        body["format"] = fmt
    return body


def _tool_calls(message: dict[str, Any], start: int) -> list[ToolCall]:
    calls = []
    for offset, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        calls.append(ToolCall(
            str(call.get("id") or f"call_{start + offset}"),
            str(function.get("name") or ""),
            arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
        ))
    return calls


def _finish(done_reason: str, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return FINISH_TOOL_CALLS
    return FINISH_LENGTH if done_reason == "length" else FINISH_STOP


def _usage(payload: dict[str, Any]) -> Usage:
    return Usage(int(payload.get("prompt_eval_count") or 0), int(payload.get("eval_count") or 0))


def from_wire(payload: dict[str, Any], model: str) -> ChatResult:
    if payload.get("error"):
        raise GatewayError(502, f"upstream: {payload['error']}")
    message = payload.get("message") or {}
    calls = _tool_calls(message, 0)
    return ChatResult(
        id="",
        model=str(payload.get("model") or model),
        text=str(message.get("content") or ""),
        tool_calls=calls,
        finish_reason=_finish(str(payload.get("done_reason") or ""), bool(calls)),
        usage=_usage(payload),
    )


def stream_events(lines: Iterable[bytes]) -> Iterator[StreamEvent]:
    """Canonical events from Ollama's NDJSON stream."""
    tool_count = 0
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if chunk.get("error"):
            raise GatewayError(502, f"upstream: {chunk['error']}")
        message = chunk.get("message") or {}
        if message.get("content"):
            yield StreamEvent("text", text=str(message["content"]))
        # Ollama sends each tool call whole, in one chunk.
        for call in _tool_calls(message, tool_count):
            yield StreamEvent("tool_call", index=tool_count, tool_id=call.id, tool_name=call.name,
                              arguments=call.arguments)
            tool_count += 1
        if chunk.get("done"):
            yield StreamEvent("usage", usage=_usage(chunk))
            yield StreamEvent("finish", finish_reason=_finish(str(chunk.get("done_reason") or ""), tool_count > 0))
            return
    yield StreamEvent("finish", finish_reason=FINISH_TOOL_CALLS if tool_count else FINISH_STOP)
