"""Backward-compat re-exports — prefer importing from dalal_agents.agents.analysts."""
from dalal_agents.agents.analysts import (
    SentimentAnalystAgent,
    NewsAnalystAgent,
    FundamentalsAnalystAgent,
    AnalystTeam,
)

__all__ = [
    "SentimentAnalystAgent",
    "NewsAnalystAgent",
    "FundamentalsAnalystAgent",
    "AnalystTeam",
]
