from __future__ import annotations

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.agents.trading.helpers import (
    _analyst_signal_table,
    _research_debate_summary,
    _risk_assessment_summary,
    _risk_debate_summary,
    _trade_proposal_summary,
)
from dalal_agents.models import FinalDecision, TradingState

_FUND_MANAGER_SYSTEM = """\
You are the Chief Investment Officer making the final trade decision.

You have full authority to: approve the trade, adjust size or levels, or reject (HOLD).

PHILOSOPHY:
1. Never chase a stock that has already moved 5 %+ above the original entry target.
2. Capital preservation comes before chasing returns — if doubt exists, reduce size or HOLD.
3. Think in terms of long-term compounding: a 3 % position that works is better than a
   10 % position that forces a panic exit.

PRIOR DECISIONS SECTION (if present):
  - Review prior decisions for THIS ticker.
  - If the previous call was BUY and the stock then hit our target, size can be maintained.
  - If the previous call was BUY and the stock fell to our stop-loss, reduce size by 30 %.
  - If this is the second consecutive HOLD recommendation, explicitly state why you are not
    acting rather than just repeating HOLD.

OUTPUT REQUIREMENTS:
- rationale: MUST be written in plain English that a retail investor can understand.
  No jargon: say "the stock looks cheap relative to peers" not "EV/EBITDA is at a discount".
- dissenting_view: capture the single strongest argument from the LOSING side of the
  research debate — even if you overruled it.
- action, entry_price, target_price, stop_loss: use the risk-adjusted values.
- position_size_pct: use the risk-assessed adjusted size.
- Include ticker, exchange (NSE or BSE), and as_of_date.
"""


class FundManagerAgent(BaseAgent):

    def __init__(self, llm, memory_context: str = ""):
        super().__init__(llm)
        self._memory_context = memory_context

    @property
    def system_prompt(self) -> str:
        return _FUND_MANAGER_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        return []

    @property
    def output_schema(self) -> type[BaseModel]:
        return FinalDecision

    def _build_user_message(self, state: TradingState) -> str:
        prior_section = (
            f"\n=== PRIOR DECISIONS (MEMORY) ===\n{self._memory_context}\n"
            if self._memory_context
            else ""
        )
        return (
            f"Make the final investment decision for **{state.ticker}** ({state.exchange}) "
            f"as of **{state.analysis_date}**.\n\n"
            + prior_section +
            f"\n=== STAGE I — ANALYST SIGNALS ===\n{_analyst_signal_table(state)}\n\n"
            f"=== RESEARCH DEBATE ===\n{_research_debate_summary(state)}\n\n"
            f"=== TRADER PROPOSAL ===\n{_trade_proposal_summary(state)}\n\n"
            f"=== RISK DEBATE ===\n{_risk_debate_summary(state)}\n\n"
            f"=== RISK ASSESSMENT ===\n{_risk_assessment_summary(state)}\n\n"
            "=== YOUR FINAL DECISION ===\n"
            "Review all stages above and make the final call.\n"
            "Output a FinalDecision as JSON inside a ```json ... ``` fence.\n\n"
            "CRITICAL requirements for the JSON:\n"
            "  - rationale: write in plain English — NO jargon, NO formulas. "
            "A parent who knows nothing about investing should understand it.\n"
            "  - dissenting_view: find the strongest argument from the LOSING side "
            "of the research debate and summarise it in 1–2 sentences.\n"
            f"  - ticker: \"{state.ticker}\"\n"
            f"  - exchange: \"{state.exchange.value}\"\n"
            f"  - as_of_date: \"{state.analysis_date}\""
        )
