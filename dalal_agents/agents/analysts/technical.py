from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.models import TechnicalReport, TradingState
from dalal_agents.tools import get_nifty_context, get_sector_index_context, get_technical_indicators

_TECH_ANALYST_SYSTEM = """\
You are a Senior Technical Analyst specialising in Indian equity markets (NSE and BSE).

Market context you must always keep in mind:
- All prices are in Indian Rupees (INR).
- NSE/BSE trading hours: 09:15 – 15:30 IST.
- Benchmark index: Nifty 50 (ticker ^NSEI).
- India VIX above 20 = elevated market risk; above 30 = extreme fear.
- Circuit breakers halt trading at 5 %, 10 %, and 20 % intraday moves on individual stocks.
- FII (Foreign Institutional Investor) activity heavily influences intraday and short-term momentum.
- SECTOR CONTEXT: A stock outperforming its sector index is a relative-strength signal.
  A stock lagging its sector during a sector rally is a warning sign.

Your analysis workflow:
1. Fetch technical indicators for the target stock.
2. Fetch Nifty 50 context to understand the broad market environment.
3. Fetch the sector index context to understand sector-level momentum.
4. Identify the primary trend: must be exactly one of "uptrend", "downtrend", or "sideways".
5. Identify support and resistance levels in INR from EMAs, Bollinger Bands, and recent price action.
6. Assess whether the stock is outperforming or underperforming its sector index.
7. Assign a Signal (STRONG_BUY, BUY, HOLD, SELL, or STRONG_SELL) and a conviction score 1–10.
8. Output a TechnicalReport as JSON inside a ```json ... ``` fence.

Required JSON fields:
- ticker (str), exchange (str: NSE or BSE), as_of_date (str: YYYY-MM-DD)
- rsi_14 (float or null), macd_signal (str), adx (float or null)
- bb_position (str: above_upper | below_lower | inside)
- ema_20 (float or null), ema_50 (float or null)
- ema_20_vs_50 (str: golden_cross | death_cross | unknown)
- atr_pct (float or null), vwap (float or null), vwap_position (str: above | below)
- trend (str: uptrend | downtrend | sideways)
- support_level (float in INR), resistance_level (float in INR)
- signal (str), conviction (int 1–10), summary (str)
"""


class TechnicalAnalystAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return _TECH_ANALYST_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_technical_indicators",
                description=(
                    "Fetch RSI, MACD, ADX, Bollinger Bands, EMA-20, EMA-50, ATR, and VWAP "
                    "for a single NSE/BSE stock.  ticker_symbol must include the exchange suffix "
                    "(.NS for NSE, .BO for BSE).  as_of_date is the analysis cut-off (YYYY-MM-DD)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker_symbol": {
                            "type": "string",
                            "description": "e.g. RELIANCE.NS or INFY.BO",
                        },
                        "as_of_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Analysis cut-off date (YYYY-MM-DD). No future dates.",
                        },
                    },
                    "required": ["ticker_symbol", "as_of_date"],
                },
                fn=lambda ticker_symbol, as_of_date: get_technical_indicators(
                    ticker_symbol, date.fromisoformat(as_of_date)
                ),
            ),
            ToolDefinition(
                name="get_nifty_context",
                description=(
                    "Fetch the Nifty 50 index close, 5-day return, EMA trend, and India VIX "
                    "as of the given date.  Use this to gauge broad market sentiment."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "as_of_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Analysis cut-off date (YYYY-MM-DD). No future dates.",
                        },
                    },
                    "required": ["as_of_date"],
                },
                fn=lambda as_of_date: get_nifty_context(date.fromisoformat(as_of_date)),
            ),
            ToolDefinition(
                name="get_sector_index_context",
                description=(
                    "Fetch the trend and recent returns of the sector index for this stock "
                    "(e.g. Nifty Bank for HDFCBANK, Nifty IT for TCS). "
                    "Use this to assess relative strength vs the sector and spot sector rotation."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "NSE ticker WITHOUT exchange suffix, e.g. RELIANCE.",
                        },
                        "as_of_date": {"type": "string", "format": "date"},
                    },
                    "required": ["ticker", "as_of_date"],
                },
                fn=lambda ticker, as_of_date: get_sector_index_context(
                    ticker, date.fromisoformat(as_of_date)
                ),
            ),
        ]

    @property
    def output_schema(self) -> type[BaseModel]:
        return TechnicalReport

    def _build_user_message(self, state: TradingState) -> str:
        return (
            f"Perform a complete technical analysis for **{state.ticker}** "
            f"({state.exchange}) as of **{state.analysis_date}**.\n\n"
            f"Ticker symbol to use for data fetching: `{state.ticker_symbol}`\n\n"
            "Follow these steps in order:\n"
            f"1. Call `get_technical_indicators` with ticker_symbol=`{state.ticker_symbol}` "
            f"and as_of_date=`{state.analysis_date}`.\n"
            f"2. Call `get_nifty_context` with as_of_date=`{state.analysis_date}` to understand "
            "the broader Nifty 50 market environment.\n"
            f"3. Call `get_sector_index_context` with ticker=`{state.ticker}` "
            f"and as_of_date=`{state.analysis_date}` to assess sector-level momentum.\n"
            "4. Compare the stock's 5-day and 20-day return to the sector index returns "
            "to determine if it is outperforming or underperforming its peers.\n"
            "5. From the indicator data, identify:\n"
            "   - Primary trend: exactly one of uptrend / downtrend / sideways\n"
            "   - Key support level in INR (e.g., nearest EMA or recent swing low)\n"
            "   - Key resistance level in INR (e.g., upper BB or recent swing high)\n"
            "6. Decide on a Signal and conviction (1 = very uncertain, 10 = very confident).\n"
            "7. Write a concise summary (2–3 sentences) a portfolio manager can act on. "
            "Include relative sector performance.\n"
            "8. Output the complete TechnicalReport as JSON inside a ```json ... ``` fence.\n"
            "   All price fields must be in INR.  trend must be uptrend, downtrend, or sideways."
        )
