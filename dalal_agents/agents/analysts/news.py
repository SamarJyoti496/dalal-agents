from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.models import NewsReport, TradingState
from dalal_agents.tools import (
    get_corporate_actions,
    get_google_news_headlines,
    get_india_stock_news,
    get_nifty_context,
)

_NEWS_SYSTEM = """\
You are a Macro & News Analyst for Indian equity markets.

India-specific macro factors you must track:

1. RBI MONETARY POLICY.
   The Reserve Bank of India holds 6 Monetary Policy Committee (MPC) meetings per year.
   Rate hikes hurt banking stocks and NBFCs (higher cost of funds); rate cuts boost them.
   Hawkish language ("withdrawal of accommodation") is bearish for rate-sensitive sectors.

2. UNION BUDGET (February 1 each year).
   The single most market-moving event in the Indian calendar.
   Key beneficiary sectors vary by year: infrastructure, defence, pharma, EVs, railways.
   Check if the analysis date is within 30 days of Feb 1 — if so, budget impact is elevated.

3. OIL PRICES & RUPEE.
   India imports roughly 85 % of its crude oil.
   Rising oil → higher inflation → RBI hawkish → bad for bonds and leveraged companies.
   Rising oil also weakens the INR, which HURTS oil marketing companies (OMCs: HPCL, BPCL,
   IOC) but HELPS IT exporters (Infosys, TCS, Wipro) because their USD revenues convert
   to more INR.

4. MONSOON.
   June–September monsoon determines kharif crop output.
   Below-normal monsoon → rural demand falls → FMCG and two-wheeler stocks underperform.

5. CORPORATE ACTIONS CHANGE THE RISK PROFILE.
   An upcoming board meeting (results date) within 30 days increases short-term uncertainty.
   Ex-dividend or bonus dates affect price continuity — note these prominently.
   Set earnings_upcoming=true if a board meeting for results is due within 30 days.

6. OUTPUT FORMAT.
   Identify the RBI stance (hawkish / dovish / neutral), any budget or oil impact,
   sector-specific policy, upcoming earnings, and output a NewsReport JSON.
"""


class NewsAnalystAgent(BaseAgent):

    def __init__(self, llm, newsapi_key: str = ""):
        super().__init__(llm)
        self._newsapi_key = newsapi_key

    @property
    def system_prompt(self) -> str:
        return _NEWS_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        newsapi_key = self._newsapi_key
        return [
            ToolDefinition(
                name="get_india_stock_news",
                description=(
                    "Fetch recent English-language news from NewsAPI.  Call this twice: "
                    "once with a macro query (e.g. 'India RBI oil budget economy') and once "
                    "with the company name for company-specific news."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "company_name": {
                            "type": "string",
                            "description": "Company name or macro query string.",
                        },
                        "as_of_date": {"type": "string", "format": "date"},
                        "lookback_days": {"type": "integer", "default": 7},
                    },
                    "required": ["company_name", "as_of_date"],
                },
                fn=lambda company_name, as_of_date, lookback_days=7: get_india_stock_news(
                    company_name, date.fromisoformat(as_of_date), int(lookback_days), newsapi_key
                ),
            ),
            ToolDefinition(
                name="get_google_news_headlines",
                description=(
                    "Fetch free headlines from Google News RSS — no API key needed. "
                    "Prioritises Indian sources: Economic Times, Moneycontrol, Business Standard, "
                    "Mint, NDTV Profit, CNBCTV18. "
                    "Use this for company-specific news even when NEWSAPI_KEY is not configured."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query, e.g. company name or macro topic.",
                        },
                        "as_of_date": {"type": "string", "format": "date"},
                        "lookback_days": {"type": "integer", "default": 7},
                    },
                    "required": ["query", "as_of_date"],
                },
                fn=lambda query, as_of_date, lookback_days=7: get_google_news_headlines(
                    query, date.fromisoformat(as_of_date), int(lookback_days)
                ),
            ),
            ToolDefinition(
                name="get_corporate_actions",
                description=(
                    "Fetch upcoming corporate actions from NSE: dividends, bonuses, splits, "
                    "rights issues, and board meetings (results dates). "
                    "An upcoming results board meeting within 30 days should set earnings_upcoming=true."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "NSE ticker without suffix."},
                        "as_of_date": {"type": "string", "format": "date"},
                        "lookforward_days": {"type": "integer", "default": 45},
                    },
                    "required": ["ticker", "as_of_date"],
                },
                fn=lambda ticker, as_of_date, lookforward_days=45: get_corporate_actions(
                    ticker, date.fromisoformat(as_of_date), int(lookforward_days)
                ),
            ),
            ToolDefinition(
                name="get_nifty_context",
                description=(
                    "Fetch Nifty 50 close, 5-day return, EMA trend, and India VIX. "
                    "Use this to frame the stock-level news within broad market conditions."
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
        return NewsReport

    def _build_user_message(self, state: TradingState) -> str:
        return (
            f"Perform a macro and news analysis for **{state.ticker}** ({state.exchange}) "
            f"as of **{state.analysis_date}**.\n\n"
            "Follow these steps in order:\n"
            f"1. Call `get_nifty_context` with as_of_date=`{state.analysis_date}` "
            "to understand broad market conditions.\n"
            f"2. Call `get_corporate_actions` with ticker=`{state.ticker}` and "
            f"as_of_date=`{state.analysis_date}`. "
            "Flag any upcoming board meeting (results), ex-dividend, or bonus dates.\n"
            f"3. Call `get_google_news_headlines` with query=`{state.ticker} stock India` "
            f"and as_of_date=`{state.analysis_date}` for company-specific Indian news.\n"
            "4. Call `get_google_news_headlines` with query=`India RBI budget oil economy` "
            f"and as_of_date=`{state.analysis_date}` for macro-level news.\n"
            + (
                f"5. Call `get_india_stock_news` with company_name=`{state.ticker}` "
                f"and as_of_date=`{state.analysis_date}` for additional coverage.\n"
                if self._newsapi_key
                else "5. Skip NewsAPI — key not configured.\n"
            )
            + "6. From all sources, assess:\n"
            "   - rbi_stance: hawkish | dovish | neutral (or null if no RBI news)\n"
            "   - budget_impact: any Budget-related tailwinds or headwinds (or null)\n"
            "   - sebi_news: any SEBI regulatory actions (or null)\n"
            "   - sector_policy: government policy affecting the stock's sector (or null)\n"
            "   - earnings_upcoming: true if a board meeting for results is within 30 days\n"
            "   - top_headlines: the 3–5 most market-relevant headlines as strings\n"
            "7. Output a NewsReport as JSON inside a ```json ... ``` fence."
        )
