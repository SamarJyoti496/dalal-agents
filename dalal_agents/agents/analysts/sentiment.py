from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel

from dalal_agents.agents.base import BaseAgent, ToolDefinition
from dalal_agents.models import SentimentReport, TradingState
from dalal_agents.tools import (
    get_bulk_block_deals,
    get_fii_dii_flows,
    get_india_reddit_sentiment,
    get_india_stock_news,
    get_nse_options_pcr,
)

_SENTIMENT_SYSTEM = """\
You are a Sentiment Analyst specialising in Indian equity markets.

India-specific sentiment rules you must always apply:

1. FII/DII FLOWS ARE THE PRIMARY SIGNAL.
   Sustained FII selling overrides any amount of positive retail or Reddit sentiment.
   Conversely, FII net buying above ₹1,000 Cr/day for 3 or more consecutive days is a
   strong bullish institutional signal that typically precedes sustained rallies.

2. OPTIONS PCR (PUT-CALL RATIO) IS THE SECOND-MOST IMPORTANT SIGNAL.
   PCR > 1.2 means heavy put buying — the market is hedging or expecting a fall (bearish).
   PCR < 0.7 means call buying dominates — bullish sentiment, but also complacency risk.
   PCR 0.7–1.2 is neutral.  Only applies to F&O-eligible stocks.

3. BULK / BLOCK DEALS REVEAL INSTITUTIONAL CONVICTION.
   A promoter or insider buying in bulk is a strong bullish signal.
   A large FII selling in a bulk deal warrants caution even if headline flows look positive.

4. INSTITUTIONAL vs RETAIL.
   DII (domestic institutions — LIC, mutual funds) often buy on dips and provide a floor.
   If FII is selling but DII is buying aggressively, that limits downside.

5. REDDIT/SOCIAL MEDIA CONTEXT.
   The key Indian investing subreddits are r/IndiaInvestments, r/DalalStreet, and
   r/IndianStockMarket.  Treat Reddit sentiment as a secondary confirmation, not a primary driver.

6. OUTPUT FORMAT.
   Synthesise flows, PCR, bulk/block deals, news, and social sentiment into a SentimentReport JSON:
   - fii_dii_flow: a one-line plain-English description
   - overall_sentiment: "bullish", "bearish", or "neutral"
   - signal: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
   - conviction: 1–10
"""


class SentimentAnalystAgent(BaseAgent):

    def __init__(self, llm, newsapi_key: str = "", reddit_creds: Optional[dict] = None):
        super().__init__(llm)
        self._newsapi_key  = newsapi_key
        self._reddit_creds = reddit_creds or {}

    @property
    def system_prompt(self) -> str:
        return _SENTIMENT_SYSTEM

    @property
    def tools(self) -> list[ToolDefinition]:
        newsapi_key  = self._newsapi_key
        reddit_creds = self._reddit_creds

        return [
            ToolDefinition(
                name="get_fii_dii_flows",
                description=(
                    "Fetch recent FII (Foreign Institutional Investor) and DII (Domestic "
                    "Institutional Investor) net buy/sell flows in crores INR from NSE data. "
                    "This is the most important sentiment indicator for Indian markets."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "as_of_date":    {"type": "string", "format": "date",
                                          "description": "Cut-off date (YYYY-MM-DD)."},
                        "lookback_days": {"type": "integer", "default": 5,
                                          "description": "Number of trading days to fetch (default 5)."},
                    },
                    "required": ["as_of_date"],
                },
                fn=lambda as_of_date, lookback_days=5: get_fii_dii_flows(
                    date.fromisoformat(as_of_date), int(lookback_days)
                ),
            ),
            ToolDefinition(
                name="get_india_stock_news",
                description=(
                    "Fetch recent English-language news articles about a company or topic "
                    "from NewsAPI.  Use the full company name (e.g. 'Infosys') as company_name."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "company_name":  {"type": "string",  "description": "Full company name."},
                        "as_of_date":    {"type": "string",  "format": "date"},
                        "lookback_days": {"type": "integer", "default": 7},
                    },
                    "required": ["company_name", "as_of_date"],
                },
                fn=lambda company_name, as_of_date, lookback_days=7: get_india_stock_news(
                    company_name, date.fromisoformat(as_of_date),
                    int(lookback_days), newsapi_key
                ),
            ),
            ToolDefinition(
                name="get_india_reddit_sentiment",
                description=(
                    "Search r/IndiaInvestments, r/DalalStreet, and r/IndianStockMarket for "
                    "posts about the query and return VADER sentiment scores.  Returns an "
                    "error dict if Reddit credentials are not configured — treat as neutral."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query":         {"type": "string",  "description": "Search term, e.g. ticker or company name."},
                        "as_of_date":    {"type": "string",  "format": "date"},
                        "lookback_days": {"type": "integer", "default": 7},
                    },
                    "required": ["query", "as_of_date"],
                },
                fn=lambda query, as_of_date, lookback_days=7: get_india_reddit_sentiment(
                    query, date.fromisoformat(as_of_date), int(lookback_days),
                    reddit_creds.get("client_id",     ""),
                    reddit_creds.get("client_secret", ""),
                    reddit_creds.get("user_agent",    "DalalAgents/1.0"),
                ),
            ),
            ToolDefinition(
                name="get_nse_options_pcr",
                description=(
                    "Fetch the NSE options chain for an F&O stock and compute the Put-Call Ratio. "
                    "PCR > 1.2 = bearish hedging; PCR < 0.7 = bullish complacency; 0.7–1.2 = neutral. "
                    "Use the ticker WITHOUT exchange suffix (e.g. RELIANCE, not RELIANCE.NS)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker":     {"type": "string",  "description": "NSE ticker without suffix."},
                        "as_of_date": {"type": "string",  "format": "date"},
                    },
                    "required": ["ticker", "as_of_date"],
                },
                fn=lambda ticker, as_of_date: get_nse_options_pcr(
                    ticker, date.fromisoformat(as_of_date)
                ),
            ),
            ToolDefinition(
                name="get_bulk_block_deals",
                description=(
                    "Fetch NSE bulk and block deals for the stock over the past 30 days. "
                    "Promoter or insider bulk buys are a very bullish signal; large FII bulk "
                    "sells warrant caution even if aggregate flow data looks neutral."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker":        {"type": "string",  "description": "NSE ticker without suffix."},
                        "as_of_date":    {"type": "string",  "format": "date"},
                        "lookback_days": {"type": "integer", "default": 30},
                    },
                    "required": ["ticker", "as_of_date"],
                },
                fn=lambda ticker, as_of_date, lookback_days=30: get_bulk_block_deals(
                    ticker, date.fromisoformat(as_of_date), int(lookback_days)
                ),
            ),
        ]

    @property
    def output_schema(self) -> type[BaseModel]:
        return SentimentReport

    def _build_user_message(self, state: TradingState) -> str:
        reddit_step = (
            f"4. Call `get_india_reddit_sentiment` with query=`{state.ticker}` "
            f"and as_of_date=`{state.analysis_date}` to capture retail sentiment."
            if self._reddit_creds
            else "4. Skip Reddit — credentials not configured; treat Reddit sentiment as neutral."
        )
        return (
            f"Analyze market sentiment for **{state.ticker}** ({state.exchange}) "
            f"as of **{state.analysis_date}**.\n\n"
            "Follow these steps in order:\n"
            f"1. Call `get_fii_dii_flows` with as_of_date=`{state.analysis_date}` and "
            "lookback_days=`5`.  FII/DII is your PRIMARY signal.\n"
            f"2. Call `get_nse_options_pcr` with ticker=`{state.ticker}` and "
            f"as_of_date=`{state.analysis_date}`. "
            "PCR is your SECOND signal — heavy put buying means institutions are hedging.\n"
            f"3. Call `get_bulk_block_deals` with ticker=`{state.ticker}` and "
            f"as_of_date=`{state.analysis_date}`. "
            "Flag any promoter buys or large FII block sells explicitly.\n"
            f"{reddit_step}\n"
            f"5. Call `get_india_stock_news` with company_name=`{state.ticker}` and "
            f"as_of_date=`{state.analysis_date}` to gather news sentiment.\n"
            "6. Synthesise all signals into a SentimentReport:\n"
            "   - fii_dii_flow: one-line English summary of net FII and DII activity in crores\n"
            "   - overall_sentiment: bullish | bearish | neutral\n"
            "   - Weight signals: FII flows first, PCR second, bulk deals third, news/Reddit last.\n"
            "   - Sustained FII selling overrides positive retail/Reddit sentiment.\n"
            "7. Output the SentimentReport as JSON inside a ```json ... ``` fence."
        )
