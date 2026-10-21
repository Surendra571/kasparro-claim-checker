"""
LLM Provider Abstraction.
Supports live OpenAI, Anthropic, and Gemini real function/tool calling,
alongside MockLLMProvider for offline deterministic testing.
"""

import abc
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call requested by the model."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    """Represents a normalized response from an LLM provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None


class ProviderError(Exception):
    """Base exception for LLM provider errors."""


class ProviderTimeoutError(ProviderError):
    """Raised when an LLM request exceeds the configured timeout."""


class BaseLLMProvider(abc.ABC):
    """Abstract Base Class for LLM providers."""

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = 30.0,
    ) -> ProviderResponse:
        """Sends chat messages and tools to the provider."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of provider (e.g. openai, anthropic, gemini)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    """Live OpenAI provider using real function/tool calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIProvider. Install with `pip install openai`."
            ) from e

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key must be provided via argument or OPENAI_API_KEY environment variable."
            )

        self._client = openai.OpenAI(api_key=key, base_url=base_url)
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = 30.0,
    ) -> ProviderResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as e:
            raise ProviderTimeoutError(f"OpenAI request timed out after {timeout}s: {e}") from e
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {e}") from e

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.id
                name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception as json_err:
                    raise ProviderError(
                        f"Failed to parse tool call arguments as JSON: {raw_args!r}"
                    ) from json_err
                tool_calls.append(ToolCall(call_id=call_id, name=name, arguments=args))

        return ProviderResponse(
            content=msg.content,
            tool_calls=tool_calls,
            raw_response=response,
        )


class AnthropicProvider(BaseLLMProvider):
    """Live Anthropic provider using real tool-use calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. Install with `pip install anthropic`."
            ) from e

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key must be provided via argument or ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = 30.0,
    ) -> ProviderResponse:
        import anthropic

        # Separate system prompt from messages
        system_prompt = ""
        user_assistant_messages: list[dict[str, Any]] = []

        for m in messages:
            if m["role"] == "system":
                system_prompt += m["content"] + "\n"
            elif m["role"] == "tool":
                # Convert tool output into Anthropic tool_result block
                user_assistant_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id"),
                                "content": m.get("content", ""),
                            }
                        ],
                    }
                )
            elif m["role"] == "assistant" and m.get("tool_calls"):
                # Convert assistant tool_calls to Anthropic tool_use blocks
                content_blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    content_blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    f_info = tc.get("function", {})
                    args = f_info.get("arguments", "{}")
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": f_info.get("name"),
                            "input": parsed_args,
                        }
                    )
                user_assistant_messages.append({"role": "assistant", "content": content_blocks})
            else:
                user_assistant_messages.append(m)

        # Convert tool schemas to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", {})
                anthropic_tools.append(
                    {
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    }
                )

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": user_assistant_messages,
                "max_tokens": 4096,
                "timeout": timeout,
            }
            if system_prompt:
                kwargs["system"] = system_prompt.strip()
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError as e:
            raise ProviderTimeoutError(f"Anthropic request timed out after {timeout}s: {e}") from e
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}") from e

        text_content = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        call_id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return ProviderResponse(
            content=text_content if text_content else None,
            tool_calls=tool_calls,
            raw_response=response,
        )


class MockLLMProvider(BaseLLMProvider):
    """
    Fake provider for fast, deterministic, offline unit & integration testing.
    Clearly distinguishes mock testing from live API calls.
    """

    def __init__(
        self,
        responses: list[ProviderResponse] | None = None,
        model_name: str = "mock-model-v1",
    ):
        self._queue: list[ProviderResponse] = list(responses) if responses else []
        self._model_name = model_name
        self.call_history: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def add_response(self, response: ProviderResponse) -> None:
        self._queue.append(response)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = 30.0,
    ) -> ProviderResponse:
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "timeout": timeout,
            }
        )
        if not self._queue:
            raise ProviderError("MockLLMProvider run out of queued responses.")

        return self._queue.pop(0)


def get_live_provider(
    provider_name: str | None = None,
    model: str | None = None,
) -> BaseLLMProvider:
    """
    Factory function resolving configured live provider from environment.
    Supports OPENAI_API_KEY and ANTHROPIC_API_KEY.
    """
    req_provider = (provider_name or os.getenv("LLM_PROVIDER", "")).strip().lower()

    if req_provider == "openai" or (not req_provider and os.getenv("OPENAI_API_KEY")):
        return OpenAIProvider(model=model)
    elif req_provider == "anthropic" or (not req_provider and os.getenv("ANTHROPIC_API_KEY")):
        return AnthropicProvider(model=model)

    raise ProviderError(
        "No active live LLM provider found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment or .env file."
    )
