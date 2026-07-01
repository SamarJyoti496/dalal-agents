from __future__ import annotations

from typing import Any, Optional

from dalal_agents.config import ANTHROPIC_API_KEY, DEFAULT_MODEL
from dalal_agents.llm.base import LLMResponse


class AnthropicClient:
    """Async wrapper around the Anthropic Python SDK."""

    def __init__(self, model: str = DEFAULT_MODEL):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    async def call(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        text: Optional[str] = None
        tool_calls: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        resp = LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        self._calls += 1
        self._tokens_in += resp.input_tokens
        self._tokens_out += resp.output_tokens
        return resp

    def get_stats(self) -> dict:
        return {"calls": self._calls, "tokens_in": self._tokens_in, "tokens_out": self._tokens_out}
