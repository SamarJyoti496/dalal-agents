from __future__ import annotations

from openai import AsyncOpenAI

from dalal_agents.config import GROK_API_KEY, DEFAULT_GROK_MODEL
from dalal_agents.llm.base import BaseLLMClient
from dalal_agents.llm.openai import OpenAIClient


class GrokClient(OpenAIClient):
    """
    xAI Grok — OpenAI-compatible endpoint at api.x.ai.
    Inherits all message/tool conversion from OpenAIClient; only the HTTP
    client setup differs (custom base URL and API key), so __init__ calls
    BaseLLMClient directly rather than OpenAIClient's (which would point
    the client at OpenAI instead of Grok).
    """

    _BASE_URL = "https://api.x.ai/v1"

    def __init__(self, model: str = DEFAULT_GROK_MODEL):
        BaseLLMClient.__init__(self, model)
        self._client = AsyncOpenAI(
            api_key=GROK_API_KEY,
            base_url=self._BASE_URL,
        )
