"""
pipeline/run.py — run_pipeline: wires Stage I (AnalystTeam + ResearchDebate) and
Stage II (run_trading_stage) into a single entry-point with SQLite persistence
and cache/checkpoint awareness.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text

from dalal_agents.agents.analysts import AnalystTeam
from dalal_agents.agents.debate import ResearchDebate
from dalal_agents.agents.trading import run_trading_stage
from dalal_agents.config import DEBATE_ROUNDS
from dalal_agents.memory import load_recent_decisions, save_decision
from dalal_agents.models import Exchange, TradingState
from dalal_agents.pipeline.db import (
    DB_PATH,
    init_db,
    load_checkpoint,
    load_state,
    persist_state,
    save_checkpoint,
)

_console = Console()


async def run_pipeline(
    ticker: str,
    analysis_date: date,
    llm,
    exchange: Exchange = Exchange.NSE,
    newsapi_key: str = "",
    reddit_creds: Optional[dict] = None,
    skip_if_cached: bool = True,
    resume_from_checkpoint: bool = True,
    debate_rounds: int = DEBATE_ROUNDS,
    db_path: Path = DB_PATH,
    progress_callback=None,
) -> TradingState:
    """
    Full DalalAgents pipeline for one ticker and one analysis date.

    Stage 1  — Analyst Team (4 concurrent specialists)
    Stage 2  — Research Debate (Bull vs Bear)
    Stage 3  — Trading Stage (Trader - Risk Debate - Risk Assessor - Fund Manager)

    progress_callback(event, **kwargs) receives:
      event="stage_start"   stage=str
      event="stage_done"    stage=str
      event="analyst_done"  name=str, signal=str|None, conviction=int|None, error=bool
      event="cache_hit"     ticker=str, date=str

    resume_from_checkpoint: if True and the pipeline previously crashed mid-run,
      resume from the last successfully completed stage rather than restarting.
    debate_rounds: number of Bull/Bear research debate rounds (default from config).
    """
    _quiet = progress_callback is not None

    def _cb(event: str, **kwargs) -> None:
        if progress_callback:
            progress_callback(event, **kwargs)

    def _stage_rule(n: int, label: str) -> None:
        if _quiet:
            return
        _console.print()
        _console.rule(
            Text(f"Stage {n}/3 — {label}  ·  {ticker} ({exchange.value}) {analysis_date}",
                 style="bold cyan"),
            style="cyan",
        )

    # Cache check
    if skip_if_cached:
        cached = load_state(ticker, analysis_date, db_path)
        if cached is not None and cached.final_decision is not None:
            if not _quiet:
                _console.print(
                    f"\n[bold cyan]Cache hit[/bold cyan] — {ticker} {analysis_date} "
                    f"loaded from [dim]{db_path.name}[/dim] (skipping pipeline)"
                )
            _cb("cache_hit", ticker=ticker, date=str(analysis_date))
            return cached

    init_db(db_path)
    t_start = time.perf_counter()

    # Checkpoint resume
    resumed_state, last_stage = (
        load_checkpoint(ticker, analysis_date, db_path)
        if resume_from_checkpoint
        else (None, "")
    )
    if resumed_state and last_stage:
        state = resumed_state
        if not _quiet:
            _console.print(
                f"\n[bold yellow]Checkpoint found[/bold yellow] — resuming {ticker} "
                f"{analysis_date} from after '{last_stage}'"
            )
    else:
        state = TradingState(ticker=ticker, exchange=exchange, analysis_date=analysis_date)
        last_stage = ""

    # Decision memory
    memory_context = load_recent_decisions(ticker, n=5)
    if memory_context and not _quiet:
        _console.print(f"  [dim]Memory: loaded prior decisions for {ticker}[/dim]")

    # Analyst Team
    if last_stage not in ("Analyst Team", "Research Debate", "Trading Stage"):
        _stage_rule(1, "Analyst Team")
        _cb("stage_start", stage="Analyst Team")
        team = AnalystTeam(
            llm=llm,
            newsapi_key=newsapi_key,
            reddit_creds=reddit_creds,
            on_agent_done=lambda name, signal, conviction, error: _cb(
                "analyst_done", name=name, signal=signal, conviction=conviction, error=error
            ),
        )
        await team.run(state)
        save_checkpoint(state, "Analyst Team", db_path)
        _cb("stage_done", stage="Analyst Team")
    else:
        _cb("stage_start", stage="Analyst Team")
        _cb("stage_done",  stage="Analyst Team")

    # Research Debate
    if last_stage not in ("Research Debate", "Trading Stage"):
        _stage_rule(2, "Research Debate")
        _cb("stage_start", stage="Research Debate")
        debate = ResearchDebate(llm=llm, rounds=debate_rounds)
        await debate.run(state)
        if state.research_debate and not _quiet:
            rd = state.research_debate
            _console.print(
                f"  [bold]Winner:[/bold] {rd.winning_stance}  "
                f"[bold]Signal:[/bold] {rd.consensus_signal}"
            )
        save_checkpoint(state, "Research Debate", db_path)
        rd = state.research_debate
        _cb(
            "stage_done", stage="Research Debate",
            winner=str(rd.winning_stance) if rd else None,
            signal=str(rd.consensus_signal) if rd else None,
            summary=rd.facilitator_verdict if rd else None,
        )
    else:
        _cb("stage_start", stage="Research Debate")
        _cb("stage_done",  stage="Research Debate")

    # Trading Stage (4 visible sub-stages)
    _TRADING_SUB_STAGES = ("Trader", "Risk Debate", "Risk Assessment", "Fund Manager")
    if last_stage != "Trading Stage":
        _stage_rule(3, "Trading Stage")
        await run_trading_stage(
            state, llm,
            quiet=_quiet,
            memory_context=memory_context,
            progress_callback=_cb,
        )
        save_checkpoint(state, "Trading Stage", db_path)
    else:
        for _sub in _TRADING_SUB_STAGES:
            _cb("stage_start", stage=_sub)
            _cb("stage_done",  stage=_sub)

    # Finalise, persist, and save memory
    elapsed = round(time.perf_counter() - t_start, 1)
    if state.final_decision:
        state.final_decision.pipeline_duration_seconds = elapsed
        state.final_decision.total_tool_calls = sum(
            len(r.tool_calls)
            for r in (
                state.technical_report, state.sentiment_report,
                state.news_report, state.fundamentals_report,
                state.trade_proposal, state.risk_assessment,
            )
            if r is not None
        )
        if hasattr(llm, "get_stats"):
            state.final_decision.total_llm_calls = llm.get_stats().get("calls", 0)

    persist_state(state, db_path)
    save_decision(state)          # append to dalal_memory.md
    return state