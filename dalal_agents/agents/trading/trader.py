from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.agents.trading.helpers import (
    _analyst_signal_table,
    _last_close,
    _research_debate_summary,
)
from dalal_agents.models import TradeProposal, TradingState
from dalal_agents.tools import get_nifty_context

_TRADER_SYSTEM = """\
You are a Senior Equity Trader at an India-focused proprietary trading desk.
Synthesise the analyst reports and research debate into a single CONCRETE trade recommendation.

HARD RULES — follow these without exception:
1. action must be exactly one of: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
2. entry_price, target_price, and stop_loss must be precise INR values.
3. risk_reward_ratio = (target − entry) / (entry − stop).
   If you cannot achieve R/R >= 1.5, output action=HOLD with no entry/target/stop.
4. holding_period must be exactly one of: intraday | swing_1w | positional_1m | investment_6m
5. Position sizing rules (position_size_pct):
   • Base size:  5 %
   • Boost to 10 %: ONLY if all available analysts agree AND all have conviction >= 7
     AND Nifty 50 is in a bullish trend.
   • Reduce to 2–3 %: if analyst signals are mixed OR India VIX > 20.
   • VIX > 20 cuts any final size by 30 % (round down, minimum 1 %).
6. If Nifty 50 is in a downtrend, output HOLD regardless of individual stock signals
   unless the signal is STRONG_BUY with conviction >= 9.

Use the technical support level as the stop-loss anchor and the resistance level as
the initial target. Adjust for current price momentum.
"""


class TraderAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return _TRADER_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_last_close",
                description=(
                    "Fetch the most recent closing price for a stock as of a given date. "
                    "Returns ticker_symbol, last_close_inr, and date. "
                    "Use ticker WITH exchange suffix (.NS or .BO)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker_symbol": {"type": "string",
                                          "description": "e.g. HDFCBANK.NS"},
                        "as_of_date":    {"type": "string", "format": "date"},
                    },
                    "required": ["ticker_symbol", "as_of_date"],
                },
                fn=lambda ticker_symbol, as_of_date: _last_close(ticker_symbol, as_of_date),
            ),
            ToolDefinition(
                name="get_nifty_context",
                description=(
                    "Fetch Nifty 50 trend, 5-day return, EMA positions, and India VIX. "
                    "Required to apply the position-sizing VIX and trend rules."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "as_of_date": {"type": "string", "format": "date"},
                    },
                    "required": ["as_of_date"],
                },
                fn=lambda as_of_date: get_nifty_context(date.fromisoformat(as_of_date)),
            ),
        ]

    @property
    def output_schema(self) -> type[BaseModel]:
        return TradeProposal

    def _build_user_message(self, state: TradingState) -> str:
        return (
            f"Build a trade for **{state.ticker}** ({state.exchange}) "
            f"as of **{state.analysis_date}**.\n\n"
            f"=== ANALYST TEAM SIGNALS ===\n{_analyst_signal_table(state)}\n\n"
            f"=== RESEARCH DEBATE ===\n{_research_debate_summary(state)}\n\n"
            "=== YOUR TASK ===\n"
            f"1. Call `get_last_close` with ticker_symbol=`{state.ticker_symbol}` "
            f"and as_of_date=`{state.analysis_date}`.\n"
            f"2. Call `get_nifty_context` with as_of_date=`{state.analysis_date}` "
            "(needed for VIX and trend rules).\n"
            "3. From the current price and technical levels, determine:\n"
            "   - entry_price: near current price or at a support-retest level\n"
            "   - stop_loss:   at or just below technical support\n"
            "   - target_price: near technical resistance or a reasonable risk-multiple\n"
            "   - Verify R/R = (target − entry) / (entry − stop) >= 1.5\n"
            "4. Apply position-sizing rules from your system prompt.\n"
            "5. Output a TradeProposal as JSON inside a ```json ... ``` fence.\n"
            "   holding_period must be one of: intraday | swing_1w | positional_1m | investment_6m"
        )
