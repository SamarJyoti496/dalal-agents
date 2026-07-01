from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    text: Optional[str] = None
    tool_calls: list[dict] = Field(default_factory=list)  # [{id, name, input}]
    input_tokens: int = 0
    output_tokens: int = 0
