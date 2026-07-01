"""
cli/commands.py — Command-line interface for the DalalAgents trading system.

Usage:
  python dalal.py run    TICKER [--date YYYY-MM-DD] [--exchange NSE|BSE]
  python dalal.py show   TICKER [--date YYYY-MM-DD]
  python dalal.py history TICKER
  python dalal.py list
  python dalal.py backtest TICKER --start YYYY-MM-DD --end YYYY-MM-DD [--capital N]
"""
from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import date
from pathlib import Path

from rich import box as rich_box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dotenv import load_dotenv

load_dotenv()

from dalal_agents.llm import AnthropicClient, GeminiClient, OpenAIClient, OpenRouterClient
from dalal_agents.config import (
    ANTHROPIC_API_KEY,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    GEMINI_API_KEY,
    NEWSAPI_KEY,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_SECRET,
    REDDIT_USER_AGENT,
)
from dalal_agents.models import Exchange
from dalal_agents.pipeline import DB_PATH, get_nse_trading_days, load_state, run_backtest, run_pipeline

_console = Console()


# =============================================================================
# LLM factory
# =============================================================================

def _make_llm(provider: str, model: str | None = None):
    def _key_error(var: str) -> None:
        _console.print(
            f"[bold red][ERROR][/bold red] [bold]{var}[/bold] is not set. "
            f"Add it to your [cyan].env[/cyan] file."
        )
        sys.exit(1)

    if provider == "gemini":
        if not GEMINI_API_KEY:
            _key_error("GEMINI_API_KEY")
        return GeminiClient(model=model or DEFAULT_GEMINI_MODEL)
    if provider == "openai":
        if not OPENAI_API_KEY:
            _key_error("OPENAI_API_KEY")
        return OpenAIClient(model=model or DEFAULT_OPENAI_MODEL)
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            _key_error("OPENROUTER_API_KEY")
        return OpenRouterClient(model=model or DEFAULT_OPENROUTER_MODEL)
    # default: claude
    if not ANTHROPIC_API_KEY:
        _key_error("ANTHROPIC_API_KEY")
    return AnthropicClient(model=model or DEFAULT_MODEL)


# =============================================================================
# Formatting helpers
# =============================================================================

_SIGNAL_STYLE: dict[str, str] = {
    "STRONG_BUY":  "bold green",
    "BUY":         "green",
    "HOLD":        "yellow",
    "SELL":        "red",
    "STRONG_SELL": "bold red",
}


def _fmt_inr(val) -> str:
    if val is None:
        return "—"
    return f"₹{float(val):,.2f}"


def _fmt_tokens(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def _print_final_decision(fd, stats: dict | None = None) -> None:
    action_str = str(fd.action).replace("Signal.", "")
    border = _SIGNAL_STYLE.get(action_str, "yellow")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=16)
    grid.add_column()

    grid.add_row("Action",
                 Text(action_str, style=_SIGNAL_STYLE.get(action_str, "white")))
    grid.add_row("Position size", f"{fd.position_size_pct}% of portfolio")
    grid.add_row("Entry price",   _fmt_inr(fd.entry_price))
    grid.add_row("Target price",  _fmt_inr(fd.target_price))
    grid.add_row("Stop-loss",     _fmt_inr(fd.stop_loss))
    grid.add_row("Decided at",    str(fd.decided_at)[:19])
    grid.add_row("Pipeline time", f"{fd.pipeline_duration_seconds:.0f}s")

    if fd.rationale:
        grid.add_row("", "")
        grid.add_row(Text("Rationale", style="bold"), "")
        for sentence in (fd.rationale or "").split(". "):
            if sentence.strip():
                grid.add_row("", f"  {sentence.strip()}.")

    if fd.dissenting_view:
        grid.add_row("", "")
        grid.add_row(Text("Dissenting view", style="bold"), fd.dissenting_view)

    if stats:
        grid.add_row("", "")
        grid.add_row(
            Text("LLM stats", style="dim"),
            Text(
                f"{stats['calls']} calls · "
                f"{_fmt_tokens(stats['tokens_in'])}↑ {_fmt_tokens(stats['tokens_out'])}↓ tokens",
                style="dim",
            ),
        )

    _console.print()
    _console.print(Panel(
        grid,
        title=Text(
            f"FINAL DECISION — {fd.ticker} ({fd.exchange}) {fd.as_of_date}",
            style="bold",
        ),
        border_style=border,
        box=rich_box.DOUBLE_EDGE,
        padding=(1, 2),
    ))


# =============================================================================
# COMMAND: run
# =============================================================================

def cmd_run(args: argparse.Namespace) -> None:
    exchange = Exchange(args.exchange.upper())
    analysis_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )
    reddit_creds = None
    if REDDIT_CLIENT_ID:
        reddit_creds = {
            "client_id":     REDDIT_CLIENT_ID,
            "client_secret": REDDIT_SECRET,
            "user_agent":    REDDIT_USER_AGENT or "DalalAgents/1.0",
        }

    llm = _make_llm(args.provider, getattr(args, "model", None))
    state = asyncio.run(
        run_pipeline(
            ticker=args.ticker.upper(),
            analysis_date=analysis_date,
            llm=llm,
            exchange=exchange,
            newsapi_key=NEWSAPI_KEY or "",
            reddit_creds=reddit_creds,
            skip_if_cached=not args.no_cache,
        )
    )

    if state.final_decision:
        stats = llm.get_stats() if hasattr(llm, "get_stats") else None
        _print_final_decision(state.final_decision, stats=stats)
    else:
        _console.print("[bold red][ERROR][/bold red] Pipeline completed but final_decision is None.")
        sys.exit(1)


# =============================================================================
# COMMAND: show
# =============================================================================

def cmd_show(args: argparse.Namespace) -> None:
    analysis_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )
    state = load_state(args.ticker.upper(), analysis_date)

    if state is None:
        _console.print(
            f"\n[bold red][ERROR][/bold red] No cached run found for "
            f"[bold]{args.ticker.upper()}[/bold] on [bold]{analysis_date}[/bold].\n"
            f"  Run: [cyan]python dalal.py run {args.ticker.upper()} --date {analysis_date}[/cyan]"
        )
        sys.exit(1)

    if state.final_decision:
        _print_final_decision(state.final_decision)
    else:
        _console.print("[bold yellow][WARNING][/bold yellow] No final_decision in stored state.")

    if state.research_debate:
        rd = state.research_debate
        _console.print()
        _console.rule(Text("Research Debate", style="bold magenta"), style="magenta")
        _console.print(
            f"  [bold]Winner:[/bold] {rd.winning_stance}  "
            f"[bold]Consensus:[/bold] {rd.consensus_signal}"
        )
        if rd.facilitator_verdict:
            _console.print(Panel(rd.facilitator_verdict, title="Facilitator Verdict",
                                 border_style="magenta", padding=(0, 2)))
        for turn in rd.turns:
            stance_color = "green" if str(turn.stance) == "BULLISH" else "red"
            _console.print()
            _console.print(
                f"  [bold {stance_color}]Turn {turn.turn_number}[/bold {stance_color}] "
                f"[bold]{turn.speaker}[/bold] [{turn.stance}]"
            )
            _console.print(f"  {turn.argument}")
            for pt in (turn.key_points or []):
                _console.print(f"    [dim]•[/dim] {pt}")

    _console.print()
    _console.rule(Text("Analyst Report Summaries", style="bold blue"), style="blue")
    for label, report in [
        ("Technical",    state.technical_report),
        ("Sentiment",    state.sentiment_report),
        ("News",         state.news_report),
        ("Fundamentals", state.fundamentals_report),
    ]:
        if report:
            sig = str(report.signal).replace("Signal.", "")
            style = _SIGNAL_STYLE.get(sig, "white")
            _console.print(
                f"\n  [bold]{label}[/bold] — "
                f"[{style}]{sig}[/{style}], conviction [bold]{report.conviction}[/bold]/10"
            )
            _console.print(f"  {report.summary}")
        else:
            _console.print(f"\n  [dim]{label} — NOT AVAILABLE[/dim]")
    _console.print()


# =============================================================================
# COMMAND: history
# =============================================================================

def cmd_history(args: argparse.Namespace) -> None:
    if not DB_PATH.exists():
        _console.print(f"[dim][INFO][/dim] No database found at {DB_PATH}. Run a pipeline first.")
        return

    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            """
            SELECT analysis_date, action, entry_price, target_price,
                   stop_loss, pipeline_secs
            FROM decisions
            WHERE ticker = ?
            ORDER BY analysis_date DESC
            """,
            (args.ticker.upper(),),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        _console.print(f"[dim][INFO][/dim] No history found for [bold]{args.ticker.upper()}[/bold].")
        return

    t = Table(
        title=f"History — {args.ticker.upper()}",
        box=rich_box.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold cyan",
    )
    t.add_column("Date",   style="dim",        min_width=12)
    t.add_column("Action", min_width=12)
    t.add_column("Entry",  justify="right")
    t.add_column("Target", justify="right")
    t.add_column("Stop",   justify="right")
    t.add_column("Secs",   justify="right", style="dim")

    for (dt, action, entry, target, stop, secs) in rows:
        style = _SIGNAL_STYLE.get(action, "white")
        t.add_row(
            dt,
            Text(action, style=style),
            _fmt_inr(entry),
            _fmt_inr(target),
            _fmt_inr(stop),
            f"{(secs or 0):.0f}s",
        )
    _console.print()
    _console.print(t)
    _console.print(f"  [dim]{len(rows)} record(s)[/dim]\n")


# =============================================================================
# COMMAND: list
# =============================================================================

def cmd_list(_args: argparse.Namespace) -> None:
    if not DB_PATH.exists():
        _console.print(f"[dim][INFO][/dim] No database found at {DB_PATH}. Run a pipeline first.")
        return

    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            """
            SELECT ticker, exchange, COUNT(*) as runs,
                   MIN(analysis_date) as first_run,
                   MAX(analysis_date) as last_run
            FROM decisions
            GROUP BY ticker, exchange
            ORDER BY runs DESC, ticker
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        _console.print("[dim][INFO][/dim] No analyses found. Run: [cyan]python dalal.py run RELIANCE[/cyan]")
        return

    t = Table(
        title="Analyzed Tickers",
        box=rich_box.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold cyan",
    )
    t.add_column("Ticker",   style="bold", min_width=14)
    t.add_column("Exchange", style="dim",  min_width=6)
    t.add_column("Runs",     justify="right")
    t.add_column("First",    style="dim")
    t.add_column("Last",     style="dim")

    for (ticker, exch, runs, first, last) in rows:
        t.add_row(ticker, exch, str(runs), first, last)

    _console.print()
    _console.print(t)
    _console.print(f"  [dim]{len(rows)} ticker(s)  ·  DB: {DB_PATH}[/dim]\n")


# =============================================================================
# COMMAND: backtest
# =============================================================================

def cmd_backtest(args: argparse.Namespace) -> None:
    exchange      = Exchange(args.exchange.upper())
    start_date    = date.fromisoformat(args.start)
    end_date      = date.fromisoformat(args.end)
    capital       = float(args.capital)

    if start_date >= end_date:
        _console.print("[bold red][ERROR][/bold red] --start must be before --end.")
        sys.exit(1)

    trading_days = get_nse_trading_days(start_date, end_date)
    if not trading_days:
        _console.print(
            f"[bold red][ERROR][/bold red] No NSE trading days found between {start_date} and {end_date}."
        )
        sys.exit(1)

    _console.print(
        f"\n[bold]Backtest:[/bold] {args.ticker.upper()} ({exchange.value})\n"
        f"  [bold]Period:[/bold]  {start_date} → {end_date}  ({len(trading_days)} trading days)\n"
        f"  [bold]Capital:[/bold] ₹{capital:,.0f}\n"
        f"  [dim]Results cached in {DB_PATH.name} — re-runs reuse cached decisions.[/dim]"
    )

    llm     = _make_llm(args.provider, getattr(args, "model", None))
    summary = asyncio.run(
        run_backtest(
            ticker=args.ticker.upper(),
            start_date=start_date,
            end_date=end_date,
            llm=llm,
            initial_capital=capital,
            exchange=exchange,
            newsapi_key=NEWSAPI_KEY or "",
        )
    )

    if "error" in summary:
        _console.print(f"[bold red][ERROR][/bold red] {summary['error']}")
        sys.exit(1)

    ret = summary["cumulative_return_pct"]
    ret_color = "green" if ret >= 0 else "red"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=20)
    grid.add_column()
    grid.add_row("Initial capital",   f"₹{summary['initial_capital']:,.0f}")
    grid.add_row("Final value",       f"₹{summary['final_value']:,.0f}")
    grid.add_row("Cumulative return", Text(f"{ret:+.2f}%", style=ret_color))
    grid.add_row("Sharpe ratio",      f"{summary['sharpe_ratio']:.3f}")
    grid.add_row("Max drawdown",      Text(f"{summary['max_drawdown_pct']:.2f}%", style="red"))
    grid.add_row("Trading days",      str(summary["trading_days"]))
    grid.add_row("Run ID",            Text(summary["run_id"], style="dim"))

    _console.print()
    _console.print(Panel(
        grid,
        title=Text(f"Backtest Results — {args.ticker.upper()}", style="bold yellow"),
        border_style="yellow",
        box=rich_box.ROUNDED,
        padding=(1, 2),
    ))


# =============================================================================
# Argument parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dalal",
        description="DalalAgents — Multi-agent LLM trading system for Indian markets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python dalal.py run    RELIANCE --date 2024-01-15\n"
            "  python dalal.py show   RELIANCE --date 2024-01-15\n"
            "  python dalal.py history RELIANCE\n"
            "  python dalal.py list\n"
            "  python dalal.py backtest TCS --start 2024-01-01 --end 2024-03-31\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the full pipeline for one ticker and date.")
    p_run.add_argument("ticker",                          help="NSE/BSE ticker symbol (e.g. RELIANCE)")
    p_run.add_argument("--date",     default=None,        help="Analysis date YYYY-MM-DD (default: today)")
    p_run.add_argument("--exchange", default="NSE",       help="NSE or BSE (default: NSE)")
    p_run.add_argument("--no-cache", action="store_true", help="Force re-run even if cached")
    p_run.add_argument(
        "--provider", default="claude", choices=["claude", "gemini", "openai", "openrouter"],
        help="LLM provider: claude (default), gemini, openai, or openrouter",
    )
    p_run.add_argument("--model", default=None, help="Override default model name for the chosen provider")

    # show
    p_show = sub.add_parser("show", help="Show a stored pipeline result in detail.")
    p_show.add_argument("ticker",                   help="NSE/BSE ticker symbol")
    p_show.add_argument("--date", default=None,     help="Analysis date YYYY-MM-DD (default: today)")

    # history
    p_hist = sub.add_parser("history", help="Show all past decisions for a ticker.")
    p_hist.add_argument("ticker", help="NSE/BSE ticker symbol")

    # list
    sub.add_parser("list", help="List all tickers that have been analyzed.")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run a historical backtest over a date range.")
    p_bt.add_argument("ticker",                          help="NSE/BSE ticker symbol")
    p_bt.add_argument("--start",    required=True,       help="Start date YYYY-MM-DD")
    p_bt.add_argument("--end",      required=True,       help="End date YYYY-MM-DD")
    p_bt.add_argument("--capital",  default=1_000_000,   help="Starting capital in INR (default: 1000000)")
    p_bt.add_argument("--exchange", default="NSE",       help="NSE or BSE (default: NSE)")
    p_bt.add_argument(
        "--provider", default="claude", choices=["claude", "gemini", "openai", "openrouter"],
        help="LLM provider: claude (default), gemini, openai, or openrouter",
    )
    p_bt.add_argument("--model", default=None, help="Override default model name for the chosen provider")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "run":      cmd_run,
        "show":     cmd_show,
        "history":  cmd_history,
        "list":     cmd_list,
        "backtest": cmd_backtest,
    }
    dispatch[args.command](args)
