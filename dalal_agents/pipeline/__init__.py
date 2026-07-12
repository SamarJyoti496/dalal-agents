"""
dalal_agents/pipeline — Final orchestration layer for DalalAgents.

Split across:
  db.py        SQLite schema + init_db / persist_state / load_state / checkpoints
  run.py       run_pipeline — Stage I/II/III orchestration
  calendar.py  get_nse_trading_days — real NSE trading calendar via Nifty 50
  backtest.py  run_backtest — historical simulation
"""

from dalal_agents.pipeline.db import (
    DB_PATH,
    init_db,
    load_checkpoint,
    load_state,
    persist_state,
    save_checkpoint,
)
from dalal_agents.pipeline.run import run_pipeline
from dalal_agents.pipeline.calendar import get_nse_trading_days
from dalal_agents.pipeline.backtest import run_backtest

__all__ = [
    "DB_PATH",
    "init_db",
    "persist_state",
    "load_state",
    "save_checkpoint",
    "load_checkpoint",
    "run_pipeline",
    "get_nse_trading_days",
    "run_backtest",
]


if __name__ == "__main__":
    import asyncio
    from datetime import date

    from rich.console import Console
    from rich.text import Text

    from dalal_agents.llm import AnthropicClient

    _console = Console()
    llm = AnthropicClient()
    state = asyncio.run(run_pipeline("RELIANCE", date(2024, 1, 15), llm))

    fd = state.final_decision
    if fd:
        _console.rule(Text("Final Decision", style="bold"), style="bright_white")
        _console.print(f"  Action: [bold]{fd.action}[/bold]  ·  {fd.ticker}  {fd.as_of_date}")
        _console.print(
            f"  Entry ₹{fd.entry_price}  Target ₹{fd.target_price}  Stop ₹{fd.stop_loss}"
        )
        _console.print(f"  {fd.rationale}")
