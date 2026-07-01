from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "")   # defaults to localhost:11434/v1
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET: str = os.getenv("REDDIT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "")

DEFAULT_MODEL: str = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL: str = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL: str = "gpt-4o"
DEFAULT_OPENROUTER_MODEL: str = "openai/gpt-oss-20b:free"
DEFAULT_GROK_MODEL: str = "grok-2-latest"
DEFAULT_OLLAMA_MODEL: str = "qwen2.5:14b"
MAX_TOOL_ITERATIONS: int = 8
DEBATE_ROUNDS: int = 2
RISK_DEBATE_ROUNDS: int = 1
MAX_POSITION_SIZE_PCT: float = 10.0
MIN_RISK_REWARD_RATIO: float = 1.5
