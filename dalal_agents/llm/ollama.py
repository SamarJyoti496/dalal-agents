from __future__ import annotations

from openai import AsyncOpenAI

from dalal_agents.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL
from dalal_agents.llm.base import BaseLLMClient
from dalal_agents.llm.openai import OpenAIClient


class OllamaClient(OpenAIClient):
    """
    Ollama local inference server — OpenAI-compatible API at localhost:11434.

    Run any GGUF model locally (Qwen2.5, Mistral, LLaMA 3, etc.) with zero
    API cost.  Start the server with `ollama serve` and pull a model with
    `ollama pull qwen2.5:14b` before running the pipeline.

    OLLAMA_BASE_URL defaults to http://localhost:11434/v1 but can be
    overridden in .env for remote Ollama deployments.

    __init__ calls BaseLLMClient directly rather than OpenAIClient's (which
    would point the client at OpenAI instead of the local Ollama server).
    """

    _DEFAULT_BASE = "http://localhost:11434/v1"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL):
        BaseLLMClient.__init__(self, model)
        self._client = AsyncOpenAI(
            api_key="ollama",  # Ollama ignores the key
            base_url=OLLAMA_BASE_URL or self._DEFAULT_BASE,
        )
