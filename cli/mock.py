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
    from dalal_agents.models import Exchange, FinalDecision, Signal, TradingState

    _analysts = [
        ("Technical",    "BUY",        8),
        ("Sentiment",    "HOLD",        5),
        ("News",         "STRONG_BUY",  9),
        ("Fundamentals", "BUY",         7),
    ]

    callback("stage_start", stage="Analyst Team")
    for name, signal, conviction in _analysts:
        await asyncio.sleep(0.7)
        callback("analyst_done", name=name, signal=signal, conviction=conviction, error=False)
    await asyncio.sleep(0.3)
    callback("stage_done", stage="Analyst Team")

    callback("stage_start", stage="Research Debate")
    await asyncio.sleep(1.2)
    callback("stage_done", stage="Research Debate")

    for _sub in ("Trader", "Risk Debate", "Risk Assessment", "Fund Manager"):
        callback("stage_start", stage=_sub)
        await asyncio.sleep(0.4)
        callback("stage_done", stage=_sub)

    exch = Exchange(exchange_str)
    state = TradingState(
        ticker=ticker,
        exchange=exch,
        analysis_date=datetime.date.fromisoformat(date_str),
    )
    state.final_decision = FinalDecision(
        ticker=ticker,
        exchange=exch,
        as_of_date=datetime.date.fromisoformat(date_str),
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
    return state
