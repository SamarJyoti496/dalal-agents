from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from google import genai

from dalal_agents.config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY
from dalal_agents.llm.base import LLMResponse


class GeminiClient:
    """
    Async wrapper around the Google Generative AI SDK (google-genai >= 1.0).

    Translates between the Anthropic content-block message format used by
    BaseAgent and the Gemini Content / Part format expected by the SDK.
    """

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    @staticmethod
    def _to_gemini_tools(tools: list[dict]):
        from google.genai import types
        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("input_schema", {}),
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=decls)]

    @staticmethod
    def _build_id_name_map(messages: list[dict]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        mapping[block["id"]] = block["name"]
        return mapping

    def _to_gemini_contents(self, messages: list[dict]):
        from google.genai import types

        id_to_name = self._build_id_name_map(messages)
        contents = []

        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            parts = []

            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        parts.append(types.Part.from_text(text=block["text"]))
                    elif btype == "tool_use":
                        parts.append(types.Part.from_function_call(
                            name=block["name"],
                            args=block.get("input") or {},
                        ))
                    elif btype == "tool_result":
                        fn_name = id_to_name.get(block["tool_use_id"], "unknown_tool")
                        raw = block.get("content", "{}")
                        try:
                            response_data = json.loads(raw) if isinstance(raw, str) else raw
                        except (json.JSONDecodeError, TypeError):
                            response_data = {"result": str(raw)}
                        parts.append(types.Part.from_function_response(
                            name=fn_name,
                            response=response_data,
                        ))

            if parts:
                contents.append(types.Content(role=role, parts=parts))

        return contents

    async def call(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        from google.genai import types

        contents = self._to_gemini_contents(messages)

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if tools:
            config_kwargs["tools"] = self._to_gemini_tools(tools)

        config = types.GenerateContentConfig(**config_kwargs)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            ),
        )

        text: Optional[str] = None
        tool_calls: list[dict] = []

        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content:
            for i, part in enumerate(candidate.content.parts):
                if getattr(part, "text", None):
                    text = part.text
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append({
                        "id": f"gemini_{fc.name}_{i}",
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    })

        usage = getattr(response, "usage_metadata", None)
        resp = LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
        self._calls += 1
        self._tokens_in += resp.input_tokens
        self._tokens_out += resp.output_tokens
        return resp

    def get_stats(self) -> dict:
        return {"calls": self._calls, "tokens_in": self._tokens_in, "tokens_out": self._tokens_out}
