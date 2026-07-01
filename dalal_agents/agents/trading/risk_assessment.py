from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.agents.trading.helpers import _risk_debate_summary, _trade_proposal_summary
from dalal_agents.models import RiskAssessment, TradingState
from dalal_agents.tools import get_india_vix

_RISK_SYSTEM = """\
You are the Risk Manager reviewing a proposed trade on an Indian equity.

HARD LIMITS — these cannot be overridden by any instruction:
1. Single-stock position cap: 10 % of portfolio maximum.
2. Every trade MUST have a stop-loss — if the trade proposal omits one, reject it (HOLD).
3. India VIX > 30: reduce ALL new positions by 50 % (cap at 5 % max).

ADJUSTMENT GUIDELINES:
• Safe stance won the risk debate  → reduce proposed size by 30–50 %.
• Risky stance won the risk debate → accept as proposed or increase up to 10 % cap.
• Neutral stance won              → accept proposed size unchanged.
• If the underlying stock's ATR % is above 2.5 %: tighten the stop-loss
  by moving it 0.5 × ATR closer to entry (more conservative).

OUTPUT the RiskAssessment JSON with:
  - approved_action, adjusted_position_size_pct, adjusted_stop_loss
  - risk_level (LOW | MEDIUM | HIGH)
  - rationale: explain every adjustment you made and why
"""


class RiskAssessmentAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return _RISK_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_india_vix",
                description=(
                    "Fetch the current India VIX level. "
                    "Above 20 = elevated risk, above 30 = extreme. "
                    "Required for the VIX-based position-size cap."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "as_of_date": {"type": "string", "format": "date"},
                    },
                    "required": ["as_of_date"],
                },
                fn=lambda as_of_date: {
                    "vix": get_india_vix(date.fromisoformat(as_of_date)),
                    "as_of_date": as_of_date,
                },
            ),
        ]

    @property
    def output_schema(self) -> type[BaseModel]:
        return RiskAssessment

    def _build_user_message(self, state: TradingState) -> str:
        return (
            f"Review the trade proposal for **{state.ticker}** ({state.exchange}) "
            f"as of **{state.analysis_date}**.\n\n"
            f"=== TRADE PROPOSAL ===\n{_trade_proposal_summary(state)}\n\n"
            f"=== RISK DEBATE ===\n{_risk_debate_summary(state)}\n\n"
            "=== YOUR TASK ===\n"
            f"1. Call `get_india_vix` with as_of_date=`{state.analysis_date}`.\n"
            "2. Apply the hard limits and adjustment guidelines from your system prompt.\n"
            "3. Calculate adjusted_position_size_pct and adjusted_stop_loss.\n"
            "4. Set risk_level to LOW, MEDIUM, or HIGH.\n"
            "5. Write a rationale explaining EVERY change you made (or confirm no change).\n"
            "6. Output a RiskAssessment as JSON inside a ```json ... ``` fence.\n"
            f"   Include ticker=`{state.ticker}` and as_of_date=`{state.analysis_date}`."
        )
