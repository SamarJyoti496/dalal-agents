from __future__ import annotations

from dalal_agents.models import DebateStance, DebateTranscript, TradingState
from dalal_agents.agents.debate.shared import generate_debate_turn, run_facilitator

_LLM = object


def _build_research_context(state: TradingState) -> str:
    parts = [f"STOCK: {state.ticker} ({state.exchange.value})  |  DATE: {state.analysis_date}"]

    if state.technical_report:
        r = state.technical_report
        parts.append(
            f"\n[TECHNICAL ANALYSIS — {r.signal}, conviction {r.conviction}/10]\n"
            f"Trend: {r.trend}  |  RSI-14: {r.rsi_14}  |  EMA cross: {r.ema_20_vs_50}\n"
            f"MACD: {r.macd_signal}  |  BB: {r.bb_position}  |  ADX: {r.adx}\n"
            f"Support: ₹{r.support_level}  |  Resistance: ₹{r.resistance_level}\n"
            f"Summary: {r.summary}"
        )

    if state.sentiment_report:
        r = state.sentiment_report
        parts.append(
            f"\n[SENTIMENT ANALYSIS — {r.signal}, conviction {r.conviction}/10]\n"
            f"Overall: {r.overall_sentiment}  |  FII/DII: {r.fii_dii_flow}\n"
            f"News sentiment: {r.news_sentiment_score}  |  "
            f"Reddit posts: {r.reddit_post_count}  |  score: {r.reddit_sentiment_score}\n"
            f"Summary: {r.summary}"
        )

    if state.news_report:
        r = state.news_report
        headlines = "  • " + "\n  • ".join(r.top_headlines[:5]) if r.top_headlines else "None"
        parts.append(
            f"\n[NEWS ANALYSIS — {r.signal}, conviction {r.conviction}/10]\n"
            f"RBI stance: {r.rbi_stance}  |  Earnings upcoming: {r.earnings_upcoming}\n"
            f"Budget impact: {r.budget_impact}  |  Sector policy: {r.sector_policy}\n"
            f"Top headlines:\n{headlines}\n"
            f"Summary: {r.summary}"
        )

    if state.fundamentals_report:
        r = state.fundamentals_report
        parts.append(
            f"\n[FUNDAMENTAL ANALYSIS — {r.signal}, conviction {r.conviction}/10]\n"
            f"P/E: {r.pe_ratio}  |  P/B: {r.pb_ratio}  |  EV/EBITDA: {r.ev_ebitda}\n"
            f"Market cap: ₹{r.market_cap_cr} Cr  |  ROCE: {r.roce}%  |  ROE: {r.roe}%\n"
            f"Revenue growth YoY: {r.revenue_growth_yoy}%  |  "
            f"Profit growth YoY: {r.profit_growth_yoy}%\n"
            f"Promoter holding: {r.promoter_holding_pct}%  |  "
            f"Promoter pledge: {r.promoter_pledge_pct}%\n"
            f"FII holding: {r.fii_holding_pct}%  |  DII: {r.dii_holding_pct}%\n"
            f"D/E: {r.debt_to_equity}\n"
            f"Summary: {r.summary}"
        )

    if len(parts) == 1:
        parts.append("(No analyst reports available yet.)")

    return "\n".join(parts)


_BULL_SYSTEM = """\
You are the Bull Researcher in a structured investment debate about Indian equities.
Build the STRONGEST possible case FOR taking a long position in this stock.

Draw on India-specific bullish factors wherever the data supports them:
  • Strong promoter holding with zero or negligible pledge (insiders keeping their stakes)
  • Sustained FII inflows — foreign institutions are voting with money
  • PLI scheme beneficiary or Union Budget capex tailwind
  • RBI rate-cut cycle benefiting financials, real estate, and capital-goods companies
  • Growing Indian middle class expanding the addressable market
  • Stock trading below its own historical P/E or below sector P/E
  • ROCE consistently above 15 % — proof of capital efficiency
  • Earnings growth trajectory that justifies current valuation

Be specific. Cite actual numbers (RSI values, ROCE percentages, FII flow figures).
Do not make vague claims — "fundamentals look good" is not a debate argument.
Directly rebut any bearish points already in the transcript.

Respond ONLY with a JSON object:
{"argument": "<full argument>", "key_points": ["<pt 1>", "<pt 2>", "<pt 3>"]}
"""

_BEAR_SYSTEM = """\
You are the Bear Researcher in a structured investment debate about Indian equities.
Build the STRONGEST possible case AGAINST taking a long position in this stock.

Draw on India-specific bearish factors wherever the data supports them:
  • Promoter pledge above 20 % — pledged shares can be force-sold triggering a cascade
  • Falling promoter holding quarter-over-quarter — insiders reducing their own exposure
  • Sustained FII outflows — foreign money leaving is a leading indicator of pain ahead
  • High P/E without proportionate earnings growth (expensive with no near-term catalyst)
  • Exposure to China slowdown: metals, specialty chemicals, pharma API manufacturers
  • SEBI or regulatory scrutiny risk
  • High debt-to-equity in a high-interest-rate environment
  • Rupee depreciation hurting import-dependent or USD-debt-carrying companies
  • Governance or related-party transaction concerns

Be specific. Cite actual numbers from the analyst reports.
Directly rebut any bullish points already in the transcript.

Respond ONLY with a JSON object:
{"argument": "<full argument>", "key_points": ["<pt 1>", "<pt 2>", "<pt 3>"]}
"""


class ResearchDebate:

    def __init__(self, llm: _LLM, rounds: int = 2):
        self.llm = llm
        self.rounds = rounds

    async def run(self, state: TradingState) -> DebateTranscript:
        context = _build_research_context(state)
        topic = (
            f"Should we take a long position in {state.ticker} ({state.exchange.value}) "
            f"as of {state.analysis_date}?"
        )
        transcript = DebateTranscript(topic=topic)
        turn_num = 1

        for _ in range(self.rounds):
            bull_turn = await generate_debate_turn(
                self.llm,
                "Bull Researcher",
                DebateStance.BULLISH,
                _BULL_SYSTEM,
                context,
                transcript,
                turn_num,
            )
            transcript.append(bull_turn)
            turn_num += 1

            bear_turn = await generate_debate_turn(
                self.llm,
                "Bear Researcher",
                DebateStance.BEARISH,
                _BEAR_SYSTEM,
                context,
                transcript,
                turn_num,
            )
            transcript.append(bear_turn)
            turn_num += 1

        await run_facilitator(self.llm, transcript, context)
        state.research_debate = transcript
        return transcript
