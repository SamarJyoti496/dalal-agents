"""
pipeline/db.py — SQLite persistence for DalalAgents.

Schema (decisions, full_states, checkpoints, backtest_trades) plus the
read/write helpers used by run_pipeline and run_backtest.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from dalal_agents.models import TradingState

DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "dalal_agents.db"

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
