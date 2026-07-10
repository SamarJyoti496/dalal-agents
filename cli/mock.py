"""
cli/mock.py — Fake pipeline for CLI/TUI smoke-testing.

Simulates the full 3-stage pipeline using asyncio.sleep delays so the
Rich TUI can be exercised without any LLM API calls or network access.
Import and call `fake_pipeline` from cli/main.py when running in mock mode.
"""
from __future__ import annotations

import asyncio
import datetime


async def fake_pipeline(
    ticker: str,
    date_str: str,
    exchange_str: str,
    callback,
):
    """Drive the TUI callback sequence with hardcoded fake data."""
    from dalal_agents.models import (
        DebateStance,
        DebateTranscript,
        Exchange,
        FinalDecision,
        FundamentalsReport,
        NewsReport,
        RiskAssessment,
        RiskLevel,
        SentimentReport,
        Signal,
        TechnicalReport,
        TradeProposal,
        TradingState,
    )

    as_of = datetime.date.fromisoformat(date_str)
    exch = Exchange(exchange_str)

    _analysts = [
        ("Technical",    "BUY",        8),
        ("Sentiment",    "HOLD",        5),
        ("News",         "STRONG_BUY",  9),
        ("Fundamentals", "BUY",         7),
    ]

    state = TradingState(ticker=ticker, exchange=exch, analysis_date=as_of)

    callback("stage_start", stage="Analyst Team")
    for name, signal, conviction in _analysts:
        await asyncio.sleep(0.7)
        callback("analyst_done", name=name, signal=signal, conviction=conviction, error=False)
    await asyncio.sleep(0.3)
    callback("stage_done", stage="Analyst Team")

    state.technical_report = TechnicalReport(
        ticker=ticker, exchange=exch, as_of_date=as_of,
        trend="uptrend", signal=Signal.BUY, conviction=8,
        summary="[MOCK] Price above both EMAs with a golden cross; RSI in bullish territory.",
    )
    state.sentiment_report = SentimentReport(
        ticker=ticker, as_of_date=as_of,
        overall_sentiment="mixed", signal=Signal.HOLD, conviction=5,
        summary="[MOCK] Retail chatter is neutral; institutional flows slightly positive.",
    )
    state.news_report = NewsReport(
        ticker=ticker, as_of_date=as_of,
        signal=Signal.STRONG_BUY, conviction=9,
        summary="[MOCK] Positive sector policy tailwinds and no adverse regulatory news.",
    )
    state.fundamentals_report = FundamentalsReport(
        ticker=ticker, exchange=exch, as_of_date=as_of,
        signal=Signal.BUY, conviction=7,
        summary="[MOCK] Healthy margins and manageable debt load relative to peers.",
    )

    callback("stage_start", stage="Research Debate")
    await asyncio.sleep(1.2)
    state.research_debate = DebateTranscript(
        topic=f"{ticker} — Bull vs Bear",
        facilitator_verdict="[MOCK] Bull case wins on stronger technical and news signals.",
        winning_stance=DebateStance.BULLISH,
        consensus_signal=Signal.BUY,
    )
    callback(
        "stage_done", stage="Research Debate",
        winner=str(state.research_debate.winning_stance),
        signal=str(state.research_debate.consensus_signal),
        summary=state.research_debate.facilitator_verdict,
    )

    callback("stage_start", stage="Trader")
    await asyncio.sleep(0.4)
    state.trade_proposal = TradeProposal(
        ticker=ticker, exchange=exch, as_of_date=as_of,
        action=Signal.BUY, position_size_pct=25.0,
        entry_price=2450.50, target_price=2650.00, stop_loss=2350.00,
        risk_reward_ratio=2.0, holding_period="swing_1w",
        rationale="[MOCK] Entering on breakout confirmation with a tight stop below support.",
    )
    callback(
        "stage_done", stage="Trader",
        action=str(state.trade_proposal.action),
        size=state.trade_proposal.position_size_pct,
        summary=state.trade_proposal.rationale,
    )

    callback("stage_start", stage="Risk Debate")
    await asyncio.sleep(0.4)
    state.risk_debate = DebateTranscript(
        topic=f"{ticker} — Risky vs Safe",
        facilitator_verdict="[MOCK] Position size approved with a slightly tighter stop.",
        winning_stance=DebateStance.SAFE,
        consensus_signal=Signal.BUY,
    )
    callback(
        "stage_done", stage="Risk Debate",
        winner=str(state.risk_debate.winning_stance),
        summary=state.risk_debate.facilitator_verdict,
    )

    callback("stage_start", stage="Risk Assessment")
    await asyncio.sleep(0.4)
    state.risk_assessment = RiskAssessment(
        ticker=ticker, as_of_date=as_of,
        approved_action=Signal.BUY, adjusted_position_size_pct=25.0,
        adjusted_stop_loss=2360.00, risk_level=RiskLevel.MEDIUM,
        rationale="[MOCK] Within max sector exposure limits; VIX not elevated.",
    )
    callback(
        "stage_done", stage="Risk Assessment",
        action=str(state.risk_assessment.approved_action),
        size=state.risk_assessment.adjusted_position_size_pct,
        risk_level=str(state.risk_assessment.risk_level),
        summary=state.risk_assessment.rationale,
    )

    callback("stage_start", stage="Fund Manager")
    await asyncio.sleep(0.4)
    state.final_decision = FinalDecision(
        ticker=ticker,
        exchange=exch,
        as_of_date=as_of,
        action=Signal.BUY,
        position_size_pct=25.0,
        entry_price=2450.50,
        target_price=2650.00,
        stop_loss=2350.00,
        rationale=(
            "[MOCK] Strong technical momentum confirmed by bullish sentiment. "
            "News flow is positive. Fundamentals remain solid."
        ),
        dissenting_view="[MOCK] Elevated valuations could limit near-term upside.",
        pipeline_duration_seconds=0.0,
    )
    callback(
        "stage_done", stage="Fund Manager",
        action=str(state.final_decision.action),
        size=state.final_decision.position_size_pct,
        summary=state.final_decision.rationale,
    )

    return state
