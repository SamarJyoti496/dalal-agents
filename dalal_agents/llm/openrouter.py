from __future__ import annotations

from openai import AsyncOpenAI

from dalal_agents.config import DEFAULT_OPENROUTER_MODEL, OPENROUTER_API_KEY
from dalal_agents.llm.base import BaseLLMClient
from dalal_agents.llm.openai import OpenAIClient


class OpenRouterClient(OpenAIClient):
    """
    OpenRouter (https://openrouter.ai) — OpenAI-compatible API gateway.

    Inherits all message/tool conversion from OpenAIClient; only the HTTP
    client setup differs (custom base URL, headers, and API key), so __init__
    calls BaseLLMClient directly rather than OpenAIClient's (which would
    point the client at OpenAI instead of OpenRouter).
    """

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str = DEFAULT_OPENROUTER_MODEL):
        BaseLLMClient.__init__(self, model)
        self._client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=self._BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/SamarJyoti496/dalal-agents",
                "X-Title": "DalalAgents",
            },
        )
