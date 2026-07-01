"""Backward-compat re-exports — prefer importing from dalal_agents.agents.trading."""
from dalal_agents.agents.trading import (
    TraderAgent,
    RiskAssessmentAgent,
    FundManagerAgent,
    run_trading_stage,
)

__all__ = [
    "TraderAgent",
    "RiskAssessmentAgent",
    "FundManagerAgent",
    "run_trading_stage",
]
