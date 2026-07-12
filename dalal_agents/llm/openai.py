from __future__ import annotations

import json
from typing import Any, Optional

from dalal_agents.config import OPENAI_API_KEY
from dalal_agents.llm.base import LLMResponse


class OpenAIClient:
    """Async wrapper around the OpenAI Python SDK."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        """
        Convert Anthropic content-block messages → OpenAI messages.

        Handles three block types produced by BaseAgent.run():
          tool_use    → assistant message with tool_calls array
          tool_result → role=tool message with tool_call_id
          text        → plain content string
        """
        oai: list[dict] = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                oai.append({"role": role, "content": content})
                continue

            has_tool_use = any(b.get("type") == "tool_use" for b in content if isinstance(b, dict))
            has_tool_result = any(
                b.get("type") == "tool_result" for b in content if isinstance(b, dict)
            )

            if has_tool_use:
                texts = [
                    b["text"]
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
                ]
                calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    }
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                oai.append(
                    {
                        "role": "assistant",
                        "content": texts[0] if texts else None,
                        "tool_calls": calls,
                    }
                )
            elif has_tool_result:
                for b in content:
                    if not isinstance(b, dict) or b.get("type") != "tool_result":
                        continue
                    oai.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": b.get("content", ""),
                        }
                    )
            else:
                text = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                oai.append({"role": role, "content": text})

        return oai

    async def call(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        oai_messages = self._to_openai_messages(system, messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls: list[dict] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                )

        usage = response.usage
        resp = LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        self._calls += 1
        self._tokens_in += resp.input_tokens
        self._tokens_out += resp.output_tokens
        return resp

    def get_stats(self) -> dict:
        return {"calls": self._calls, "tokens_in": self._tokens_in, "tokens_out": self._tokens_out}
