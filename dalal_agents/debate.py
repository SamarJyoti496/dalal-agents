"""Backward-compat re-exports — prefer importing from dalal_agents.agents.debate."""
from dalal_agents.agents.debate import ResearchDebate, RiskDebate, generate_debate_turn, run_facilitator
from dalal_agents.models import DebateStance, DebateTranscript, DebateTurn

__all__ = [
    "ResearchDebate",
    "RiskDebate",
    "generate_debate_turn",
    "run_facilitator",
    "DebateStance",
    "DebateTranscript",
    "DebateTurn",
]
