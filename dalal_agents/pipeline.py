"""
pipeline.py — Final orchestration layer for DalalAgents.

Wires Stage I (AnalystTeam + ResearchDebate) and Stage II (run_trading_stage)
into a single run_pipeline entry-point with SQLite persistence and cache awareness.
Includes get_nse_trading_days and run_backtest for historical simulation.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from dalal_agents.agents.analysts import AnalystTeam
from dalal_agents.agents.debate import ResearchDebate
from dalal_agents.agents.trading import run_trading_stage
from dalal_agents.config import DEBATE_ROUNDS
from dalal_agents.memory import load_recent_decisions, save_decision
from dalal_agents.models import Exchange, Signal, TradingState
from dalal_agents.tools import get_ohlcv

_console = Console()


# =============================================================================
# SECTION 1 — SQLite persistence
# =============================================================================

DB_PATH: Path = Path(__file__).resolve().parent.parent / "dalal_agents.db"

_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    exchange      TEXT    NOT NULL,
    analysis_date TEXT    NOT NULL,
    action        TEXT,
    position_size REAL,
    entry_price   REAL,
    target_price  REAL,
    stop_loss     REAL,
    rationale     TEXT,
    pipeline_secs REAL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE (ticker, analysis_date)
);

CREATE TABLE IF NOT EXISTS full_states (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    analysis_date TEXT    NOT NULL,
    state_json    TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE (ticker, analysis_date)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    analysis_date TEXT    NOT NULL,
    stage         TEXT    NOT NULL,
    state_json    TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE (ticker, analysis_date, stage)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    ticker          TEXT,
    trade_date      TEXT,
    action          TEXT,
    price           REAL,
    quantity        REAL,
    portfolio_value REAL,
    cash            REAL,
    pnl_day         REAL
);
"""

_STAGE_ORDER = ["Analyst Team", "Research Debate", "Trading Stage"]


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_DDL)
        con.commit()
    finally:
        con.close()


def persist_state(state: TradingState, db_path: Path = DB_PATH) -> None:
    if state.final_decision is None:
        return
    fd         = state.final_decision
    state_json = state.model_dump_json()

    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO decisions
                (ticker, exchange, analysis_date, action, position_size,
                 entry_price, target_price, stop_loss, rationale, pipeline_secs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.ticker,
                state.exchange.value,
                str(state.analysis_date),
                fd.action.value,
                fd.position_size_pct,
                fd.entry_price,
                fd.target_price,
                fd.stop_loss,
                (fd.rationale or "")[:1000],
                fd.pipeline_duration_seconds,
            ),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO full_states
                (ticker, analysis_date, state_json)
            VALUES (?, ?, ?)
            """,
            (state.ticker, str(state.analysis_date), state_json),
        )
        con.commit()
    finally:
        con.close()


def load_state(
    ticker: str,
    analysis_date: date,
    db_path: Path = DB_PATH,
) -> Optional[TradingState]:
    if not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "SELECT state_json FROM full_states WHERE ticker=? AND analysis_date=?",
            (ticker, str(analysis_date)),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return TradingState.model_validate_json(row[0])
    finally:
        con.close()


def save_checkpoint(state: TradingState, stage: str, db_path: Path = DB_PATH) -> None:
    """Save partial pipeline state after a stage completes (for crash recovery)."""
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO checkpoints
                (ticker, analysis_date, stage, state_json)
            VALUES (?, ?, ?, ?)
            """,
            (state.ticker, str(state.analysis_date), stage, state.model_dump_json()),
        )
        con.commit()
    finally:
        con.close()


def load_checkpoint(
    ticker: str,
    analysis_date: date,
    db_path: Path = DB_PATH,
) -> tuple[Optional[TradingState], str]:
    """
    Return (state, last_completed_stage) from the most recent checkpoint,
    or (None, '') if no checkpoint exists for this ticker/date.
    """
    if not db_path.exists():
        return None, ""
    con = sqlite3.connect(db_path)
    try:
        for stage in reversed(_STAGE_ORDER):
            cur = con.execute(
                "SELECT state_json FROM checkpoints "
                "WHERE ticker=? AND analysis_date=? AND stage=?",
                (ticker, str(analysis_date), stage),
            )
            row = cur.fetchone()
            if row:
                return TradingState.model_validate_json(row[0]), stage
    finally:
        con.close()
    return None, ""


# =============================================================================
# SECTION 2 — run_pipeline
# =============================================================================

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
    Stage 3  — Trading Stage (Trader → Risk Debate → Risk Assessor → Fund Manager)

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

    # ── Cache check ───────────────────────────────────────────────────────────
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

    # ── Checkpoint resume ─────────────────────────────────────────────────────
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

    # ── Decision memory ───────────────────────────────────────────────────────
    memory_context = load_recent_decisions(ticker, n=5)
    if memory_context and not _quiet:
        _console.print(f"  [dim]Memory: loaded prior decisions for {ticker}[/dim]")

    # ── Stage 1: Analyst Team ─────────────────────────────────────────────────
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

    # ── Stage 2: Research Debate ──────────────────────────────────────────────
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
        _cb("stage_done", stage="Research Debate")
    else:
        _cb("stage_start", stage="Research Debate")
        _cb("stage_done",  stage="Research Debate")

    # ── Stage 3: Trading Stage (4 visible sub-stages) ────────────────────────
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

    # ── Finalise, persist, and save memory ────────────────────────────────────
    elapsed = round(time.perf_counter() - t_start, 1)
    if state.final_decision:
        state.final_decision.pipeline_duration_seconds = elapsed

    persist_state(state, db_path)
    save_decision(state)          # append to dalal_memory.md
    return state


# =============================================================================
# SECTION 3 — NSE trading calendar
# =============================================================================

def get_nse_trading_days(start_date: date, end_date: date) -> list[date]:
    """
    Return actual NSE trading days between start_date and end_date (inclusive).
    Uses yfinance Nifty 50 OHLCV — weekends and exchange holidays are automatically
    absent from the index, so this gives the real trading calendar.
    """
    df = yf.download(
        "^NSEI",
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return [
        idx.date()
        for idx in df.index
        if start_date <= idx.date() <= end_date
    ]


# =============================================================================
# SECTION 4 — run_backtest
# =============================================================================

async def run_backtest(
    ticker: str,
    start_date: date,
    end_date: date,
    llm,
    initial_capital: float = 1_000_000.0,
    exchange: Exchange = Exchange.NSE,
    newsapi_key: str = "",
    db_path: Path = DB_PATH,
) -> dict:
    """
    Simulate a long-only trading strategy over a historical date range.

    For each NSE trading day the pipeline runs (or loads from cache), buys on BUY/STRONG_BUY
    signal when flat, sells on SELL/STRONG_SELL when long.
    Transaction cost: 0.2% per side.
    """
    trading_days  = get_nse_trading_days(start_date, end_date)
    run_id        = str(uuid.uuid4())
    ticker_symbol = ticker + (".NS" if exchange == Exchange.NSE else ".BO")

    _console.print()
    _console.rule(
        Text(f"Backtest  ·  {ticker} ({exchange.value})  {start_date} → {end_date}", style="bold yellow"),
        style="yellow",
    )
    _console.print(
        f"  [bold]Trading days:[/bold] {len(trading_days)}  "
        f"[bold]Capital:[/bold] ₹{initial_capital:,.0f}  "
        f"[dim]run_id={run_id}[/dim]"
    )

    cash            = initial_capital
    shares_held     = 0.0
    cost_basis      = 0.0          # average cost per share when long
    daily_values:   list[float] = []
    prev_value      = initial_capital
    TX_COST         = 0.002

    init_db(db_path)
    con = sqlite3.connect(db_path)

    try:
        for trade_date in trading_days:
            state = await run_pipeline(
                ticker, trade_date, llm, exchange, newsapi_key,
                skip_if_cached=True, db_path=db_path,
            )
            fd = state.final_decision
            if fd is None:
                continue

            # Actual close price for this trading day
            try:
                price_df  = get_ohlcv(ticker_symbol, trade_date, lookback_days=3)
                cur_price = float(price_df["Close"].iloc[-1])
            except Exception:
                cur_price = fd.entry_price or 0.0

            action   = fd.action
            quantity = 0.0

            if action in (Signal.BUY, Signal.STRONG_BUY) and shares_held == 0 and cur_price > 0:
                invest      = cash * (fd.position_size_pct / 100)
                shares_held = invest / cur_price
                cost_basis  = cur_price
                tx_cost     = invest * TX_COST
                cash       -= invest + tx_cost
                quantity    = shares_held

            elif action in (Signal.SELL, Signal.STRONG_SELL) and shares_held > 0:
                proceeds    = shares_held * cur_price
                tx_cost     = proceeds * TX_COST
                cash       += proceeds - tx_cost
                quantity    = -shares_held
                shares_held = 0.0
                cost_basis  = 0.0

            portfolio_value = cash + shares_held * cur_price
            pnl_day         = portfolio_value - prev_value
            prev_value      = portfolio_value
            daily_values.append(portfolio_value)

            con.execute(
                """
                INSERT INTO backtest_trades
                    (run_id, ticker, trade_date, action, price,
                     quantity, portfolio_value, cash, pnl_day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, ticker, str(trade_date), action.value,
                    cur_price, quantity, portfolio_value, cash, pnl_day,
                ),
            )
            con.commit()

            pnl_color = "green" if pnl_day >= 0 else "red"
            action_str = str(action.value) if hasattr(action, "value") else str(action)
            _console.print(
                f"  [dim]{trade_date}[/dim]  [bold]{action_str:<12}[/bold]  "
                f"price=[cyan]₹{cur_price:,.2f}[/cyan]  "
                f"portfolio=[cyan]₹{portfolio_value:,.0f}[/cyan]  "
                f"pnl=[{pnl_color}]₹{pnl_day:+,.0f}[/{pnl_color}]"
            )

    finally:
        con.close()

    if not daily_values:
        return {"error": "No trading days processed"}

    final_value = daily_values[-1]
    cum_return  = (final_value - initial_capital) / initial_capital * 100

    # Annualised Sharpe (risk-free = 0 for simplicity)
    returns  = np.diff(daily_values) / np.array(daily_values[:-1])
    sharpe   = float((returns.mean() / (returns.std() + 1e-9)) * (252 ** 0.5)) \
               if len(returns) >= 2 else 0.0

    # Maximum drawdown
    peak   = daily_values[0]
    max_dd = 0.0
    for v in daily_values:
        peak   = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak * 100)

    summary = {
        "run_id":               run_id,
        "ticker":               ticker,
        "start_date":           str(start_date),
        "end_date":             str(end_date),
        "trading_days":         len(trading_days),
        "initial_capital":      initial_capital,
        "final_value":          round(final_value, 2),
        "cumulative_return_pct": round(cum_return, 2),
        "sharpe_ratio":         round(sharpe, 3),
        "max_drawdown_pct":     round(max_dd, 2),
    }

    ret_color = "green" if cum_return >= 0 else "red"
    _console.print()
    _console.rule(Text("Backtest Summary", style="bold yellow"), style="yellow")
    _console.print(f"  [bold]Initial capital:[/bold]   ₹{initial_capital:,.0f}")
    _console.print(f"  [bold]Final value:[/bold]       ₹{final_value:,.0f}")
    _console.print(f"  [bold]Cumulative return:[/bold] [{ret_color}]{cum_return:.2f}%[/{ret_color}]")
    _console.print(f"  [bold]Sharpe ratio:[/bold]      {sharpe:.3f}")
    _console.print(f"  [bold]Max drawdown:[/bold]      [red]{max_dd:.2f}%[/red]")
    _console.print()

    return summary


# =============================================================================
# SECTION 5 — Main block
# =============================================================================

if __name__ == "__main__":
    import asyncio

    from dalal_agents.llm import AnthropicClient

    llm   = AnthropicClient()
    state = asyncio.run(
        run_pipeline("RELIANCE", date(2024, 1, 15), llm)
    )

    fd = state.final_decision
    if fd:
        _console.rule(Text("Final Decision", style="bold"), style="bright_white")
        _console.print(f"  Action: [bold]{fd.action}[/bold]  ·  {fd.ticker}  {fd.as_of_date}")
        _console.print(f"  Entry ₹{fd.entry_price}  Target ₹{fd.target_price}  Stop ₹{fd.stop_loss}")
        _console.print(f"  {fd.rationale}")
