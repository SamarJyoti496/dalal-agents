from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.models import FundamentalsReport, TradingState
from dalal_agents.tools import get_ohlcv, get_screener_fundamentals


def _ohlcv_tail(ticker_symbol: str, as_of_date: str, rows: int = 5) -> list[dict]:
    df = get_ohlcv(ticker_symbol, date.fromisoformat(as_of_date))
    tail = df.tail(rows)
    result = []
    for idx, row in tail.iterrows():
        result.append({
            "date":   idx.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return result


_FUNDAMENTALS_SYSTEM = """\
You are a Fundamental Analyst specialising in Indian listed companies (NSE/BSE).

India-specific fundamentals rules you must always apply:

1. ROCE OVER ROE.
   Return on Capital Employed (ROCE) is more important than ROE for Indian companies
   because it excludes leverage effects and is much harder to inflate with debt.
   Good ROCE threshold: >15 % for most sectors; >20 % for quality compounders.
   Low ROCE + high ROE = the company is borrowing heavily to generate returns → risky.

2. PROMOTER HOLDING IS CRITICAL.
   Indian companies are often family-controlled.  A falling promoter stake quarter-over-quarter
   is a significant bearish signal — insiders selling their own company.
   Promoter holding below 40 % in a traditionally promoter-driven company warrants caution.

3. PROMOTER PLEDGE IS A RED FLAG.
   Pledge percentage above 30 % is a SERIOUS RED FLAG.
   Pledged shares can be force-sold by lenders if the stock falls, creating a cascade.
   You MUST flag promoter_pledge_pct > 30 % as HIGH RISK in the summary field.

4. VALUATION IS RELATIVE.
   P/E ratio must be compared against the sector P/E, not absolute levels.
   Indian markets trade at a premium to other emerging markets — a P/E of 25x may be cheap
   for a high-quality FMCG business but expensive for a commodity stock.

5. OUTPUT FORMAT.
   All financial fields are optional floats — set to null if Screener.in doesn't provide them.
   The summary field must mention any red flags detected and a plain-English recommendation.
"""


class FundamentalsAnalystAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return _FUNDAMENTALS_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_screener_fundamentals",
                description=(
                    "Scrape Screener.in for P/E, P/B, ROCE, ROE, debt/equity, promoter holding, "
                    "promoter pledge, FII/DII holding, market cap, and more.  Use the ticker "
                    "WITHOUT exchange suffix (e.g. 'RELIANCE', not 'RELIANCE.NS')."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker":     {"type": "string",  "description": "Ticker without suffix, e.g. RELIANCE."},
                        "as_of_date": {"type": "string",  "format": "date"},
                    },
                    "required": ["ticker", "as_of_date"],
                },
                fn=lambda ticker, as_of_date: get_screener_fundamentals(
                    ticker, date.fromisoformat(as_of_date)
                ),
            ),
            ToolDefinition(
                name="get_ohlcv_tail",
                description=(
                    "Fetch the 5 most recent OHLCV bars for a stock to compute trailing price "
                    "performance.  Use the ticker WITH exchange suffix (.NS or .BO)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker_symbol": {"type": "string",  "description": "e.g. RELIANCE.NS"},
                        "as_of_date":    {"type": "string",  "format": "date"},
                    },
                    "required": ["ticker_symbol", "as_of_date"],
                },
                fn=lambda ticker_symbol, as_of_date: _ohlcv_tail(ticker_symbol, as_of_date),
            ),
        ]

    @property
    def output_schema(self) -> type[BaseModel]:
        return FundamentalsReport

    def _build_user_message(self, state: TradingState) -> str:
        return (
            f"Perform a fundamental analysis for **{state.ticker}** ({state.exchange}) "
            f"as of **{state.analysis_date}**.\n\n"
            "Follow these steps in order:\n"
            f"1. Call `get_screener_fundamentals` with ticker=`{state.ticker}` "
            f"and as_of_date=`{state.analysis_date}`.  This is your primary data source.\n"
            f"2. Call `get_ohlcv_tail` with ticker_symbol=`{state.ticker_symbol}` "
            f"and as_of_date=`{state.analysis_date}` for the last 5 days of price data.\n"
            "3. Identify and flag any of these red flags in the summary:\n"
            "   - promoter_pledge_pct > 30 % -> flag as HIGH RISK (force-sale cascade risk)\n"
            "   - promoter_holding_pct declining -> flag as bearish insider signal\n"
            "   - debt_to_equity > 2 -> flag as high leverage\n"
            "   - roce < 10 % -> flag as poor capital efficiency\n"
            "4. Assess valuation (P/E, P/B, EV/EBITDA) relative to typical sector norms.\n"
            "5. Assign a Signal and conviction based on the fundamental picture.\n"
            "6. Output a FundamentalsReport as JSON inside a ```json ... ``` fence.\n"
            "   IMPORTANT: if promoter_pledge_pct > 30, the summary MUST contain "
            "'HIGH RISK: promoter pledge above 30%'."
        )
