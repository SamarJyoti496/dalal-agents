"""
pipeline/backtest.py — run_backtest: historical simulation of a long-only
strategy driven by run_pipeline decisions, over the real NSE trading calendar.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import date
from pathlib import Path

import numpy as np

from rich.console import Console
from rich.text import Text

from dalal_agents.models import Exchange, Signal
from dalal_agents.pipeline.calendar import get_nse_trading_days
from dalal_agents.pipeline.db import DB_PATH, init_db
from dalal_agents.pipeline.run import run_pipeline
from dalal_agents.tools import get_ohlcv

_console = Console()


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
