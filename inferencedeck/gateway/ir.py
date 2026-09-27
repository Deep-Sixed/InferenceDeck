"""Canonical request/response model shared by every API and engine adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Finish reasons, in canonical form. Adapters map their API's names onto these.
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_CONTENT_FILTER = "content_filter"


class GatewayError(Exception):
    """An error with an HTTP status, rendered in the caller's API format."""

    def __init__(self, status: int, message: str, kind: str = "api_error") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        # One of: invalid_request, authentication, not_found, rate_limit,
        # overloaded, api_error. API adapters map these onto their own names.
        self.kind = kind


@dataclass
class ContentPart:
    type: str  # "text" or "image"
    text: str = ""
    # Images: either a URL (http(s) or data:) or base64 data with a media type.
    url: str = ""
    data: str = ""
    media_type: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"  # JSON text, as the model produced it


@dataclass
class Message:
    role: str  # system, user, assistant or tool
    parts: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""  # role == "tool": the call this result answers

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts if part.type == "text")


@dataclass
class Tool:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class Sampling:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class ChatRequest:
    model: str
    messages: list[Message]
    sampling: Sampling = field(default_factory=Sampling)
    tools: list[Tool] = field(default_factory=list)
    # "auto", "none", "required", or a tool name to force.
    tool_choice: str = ""
    stream: bool = False
    # OpenAI-style response_format, passed through to engines that accept it.
    response_format: dict[str, Any] | None = None
    # Engine-specific fields the client sent that have no canonical slot
    # (e.g. llama.cpp's repeat_penalty); forwarded untouched when possible.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResult:
    id: str
    model: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = FINISH_STOP
    usage: Usage = field(default_factory=Usage)


@dataclass
class StreamEvent:
    """One step of a streamed reply.

    kind is "text" (``text`` holds the delta), "tool_call" (``index`` names the
    call; ``tool_id``/``tool_name`` arrive on its first event and
    ``arguments`` holds a JSON-text delta), "usage", or "finish".
    """

    kind: str
    text: str = ""
    index: int = 0
    tool_id: str = ""
    tool_name: str = ""
    arguments: str = ""
    finish_reason: str = ""
    usage: Usage | None = None
