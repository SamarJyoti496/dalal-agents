from __future__ import annotations

from dalal_agents.config import GROK_API_KEY, DEFAULT_GROK_MODEL
from dalal_agents.llm.openai import OpenAIClient


class GrokClient(OpenAIClient):
    """
    xAI Grok — OpenAI-compatible endpoint at api.x.ai.
    Inherits all message/tool conversion from OpenAIClient.
    """

    _BASE_URL = "https://api.x.ai/v1"

    def __init__(self, model: str = DEFAULT_GROK_MODEL):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=GROK_API_KEY,
            base_url=self._BASE_URL,
        )
        self.model = model
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0
