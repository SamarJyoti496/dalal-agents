from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    text: Optional[str] = None
    tool_calls: list[dict] = Field(default_factory=list)  # [{id, name, input}]
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLMClient:
    """Shared call-count / token-usage bookkeeping for every provider client."""

    def __init__(self, model: str):
        self.model = model
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0

    def _record_usage(self, resp: LLMResponse) -> LLMResponse:
        self._calls += 1
        self._tokens_in += resp.input_tokens
        self._tokens_out += resp.output_tokens
        return resp

    def get_stats(self) -> dict:
        return {"calls": self._calls, "tokens_in": self._tokens_in, "tokens_out": self._tokens_out}
