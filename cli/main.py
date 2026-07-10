"""
cli/main.py — Interactive Rich terminal UI for DalalAgents.
Modelled after TauricResearch/TradingAgents.

Run:
    python cli/main.py
    python -m cli.main
"""
from __future__ import annotations

import asyncio
import datetime
import os
import sys
import time
from collections import deque
from pathlib import Path

import typer
import questionary
from questionary import Style as QStyle

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reconfigure stdout/stderr to UTF-8 on Windows so Rich can render Unicode box chars.
# Must happen before Console() is created.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from dalal_agents.agent import AnthropicClient, GeminiClient, OpenAIClient, OpenRouterClient
from dalal_agents.llm import GrokClient, OllamaClient
from dalal_agents.config import (
    ANTHROPIC_API_KEY, DEFAULT_MODEL,
    GEMINI_API_KEY, DEFAULT_GEMINI_MODEL,
    OPENAI_API_KEY, DEFAULT_OPENAI_MODEL,
    OPENROUTER_API_KEY, DEFAULT_OPENROUTER_MODEL,
    GROK_API_KEY, DEFAULT_GROK_MODEL,
    DEFAULT_OLLAMA_MODEL,
    NEWSAPI_KEY, REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT,
)
from dalal_agents.models import Exchange
from dalal_agents.pipeline import run_pipeline

console = Console(legacy_windows=False)

app = typer.Typer(
    name="dalal",
    help="DalalAgents: Multi-Agent LLM Trading System for Indian Markets",
    add_completion=False,
)

# ─── Questionary styles ───────────────────────────────────────────────────────

_QS_YELLOW = QStyle([
    ("selected",    "fg:yellow noinherit"),
    ("highlighted", "fg:yellow noinherit"),
    ("pointer",     "fg:yellow noinherit"),
])

_QS_CYAN = QStyle([
    ("selected",    "fg:cyan noinherit"),
    ("highlighted", "fg:cyan noinherit"),
    ("pointer",     "fg:cyan noinherit"),
])

_QS_GREEN = QStyle([
    ("text",        "fg:green"),
    ("highlighted", "noinherit"),
])

# ─── Pipeline stage → agent rows ──────────────────────────────────────────────

# Each stage maps to a list of agent display names shown in the progress table.
# Analyst Team has individual agents; trading stages are split into 4 visible rows.
_STAGE_AGENTS: dict[str, list[str]] = {
    "Analyst Team":    ["Technical Analyst", "Sentiment Analyst", "News Analyst", "Fundamentals Analyst"],
    "Research Debate": ["Research Debate"],
    "Trader":          ["Trader"],
    "Risk Debate":     ["Risk Debate"],
    "Risk Assessment": ["Risk Assessment"],
    "Fund Manager":    ["Fund Manager"],
}

_SIGNAL_COLOR = {
    "STRONG_BUY":  "bold green",
    "BUY":         "green",
    "HOLD":        "yellow",
    "SELL":        "red",
    "STRONG_SELL": "bold red",
}


# ─── State model (mirrors TradingAgents' MessageBuffer) ──────────────────────

class MessageBuffer:

    def __init__(self, max_messages: int = 100) -> None:
        self.messages:       deque                = deque(maxlen=max_messages)
        self.agent_status:   dict[str, str]       = {}
        self.analyst_detail: dict[str, str]       = {}
        self.current_report: str | None           = None
        self._start_time:    float                = time.time()

    def init(self) -> None:
        self.agent_status = {
            agent: "pending"
            for agents in _STAGE_AGENTS.values()
            for agent in agents
        }
        self.analyst_detail.clear()
        self.current_report = None
        self.messages.clear()
        self._start_time = time.time()

    def add_message(self, msg_type: str, content: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((ts, msg_type, content))

    def update_status(self, agent: str, status: str) -> None:
        if agent in self.agent_status:
            self.agent_status[agent] = status

    def agents_done(self) -> int:
        return sum(1 for s in self.agent_status.values() if s == "completed")

    def agents_total(self) -> int:
        return len(self.agent_status)


# ─── Rich layout & display ────────────────────────────────────────────────────

def create_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="main"),
        Layout(name="footer",   size=3),
    )
    layout["main"].split_column(
        Layout(name="upper",    ratio=4),   # progress (9 agent rows) + messages need most of the height
        Layout(name="analysis", ratio=1),
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=3),
        Layout(name="messages", ratio=2),
    )
    return layout


def update_display(
    layout:     Layout,
    mb:         MessageBuffer,
    ticker:     str,
    exchange:   str,
    date_str:   str,
    provider:   str,
    start_time: float,
    llm=None,
) -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    layout["header"].update(Panel(
        f"[bold green]Welcome to DalalAgents CLI[/bold green]\n"
        f"[dim]Multi-Agent LLM Trading System for Indian Markets (NSE / BSE)[/dim]",
        title=f"[bold cyan]{ticker}[/bold cyan] [dim]({exchange})[/dim]  ·  "
              f"[bold yellow]{date_str}[/bold yellow]  ·  "
              f"[bold magenta]{provider.upper()}[/bold magenta]",
        border_style="green",
        padding=(0, 2),
        expand=True,
    ))

    # ── Progress table ────────────────────────────────────────────────────────
    prog = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=True,
    )
    prog.add_column("Stage",  style="cyan",  justify="left", no_wrap=True, min_width=13)
    prog.add_column("Agent",  style="green", justify="left", no_wrap=True, min_width=15)
    prog.add_column("Status",               justify="left", no_wrap=True, min_width=18)

    for stage, agents in _STAGE_AGENTS.items():
        for i, agent in enumerate(agents):
            status    = mb.agent_status.get(agent, "pending")
            stage_lbl = stage if i == 0 else ""
            detail    = mb.analyst_detail.get(agent, "")

            if status == "in_progress":
                status_cell = Spinner("dots", text="[blue] running[/blue]", style="bold cyan")
            elif status == "completed":
                # inline signal detail so we don't need a 4th column
                status_cell = Text.from_markup(
                    f"[bold green]done[/bold green]  {detail}" if detail else "[bold green]done[/bold green]"
                )
            elif status == "error":
                status_cell = "[bold red]error[/bold red]"
            else:
                status_cell = "[yellow]pending[/yellow]"

            prog.add_row(stage_lbl, agent, status_cell)

    layout["progress"].update(
        Panel(prog, title="[bold]Progress[/bold]", border_style="cyan", padding=(0, 1))
    )

    # ── Messages panel ────────────────────────────────────────────────────────
    _TYPE_COLOR = {"System": "dim cyan", "Agent": "green", "Error": "red"}
    lines = []
    for ts, mtype, content in list(mb.messages)[-16:]:
        col = _TYPE_COLOR.get(mtype, "white")
        lines.append(f"[dim]{ts}[/dim] [{col}]{mtype:<7}[/{col}] {str(content)[:100]}")
    msg_body = "\n".join(lines) if lines else "[dim]No activity yet…[/dim]"

    layout["messages"].update(
        Panel(Text.from_markup(msg_body), title="[bold]Activity Log[/bold]",
              border_style="blue", padding=(0, 1))
    )

    # ── Analysis panel ────────────────────────────────────────────────────────
    if mb.current_report:
        layout["analysis"].update(Panel(
            Text.from_markup(mb.current_report),
            title="[bold]Current Report[/bold]",
            border_style="green",
            padding=(0, 2),
        ))
    else:
        layout["analysis"].update(Panel(
            Text.from_markup("[dim italic]Waiting for analysis report…[/dim italic]"),
            title="[bold]Current Report[/bold]",
            border_style="green",
            padding=(1, 2),
        ))

    # ── Footer stats ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    elapsed_str = f"[bold]⏱[/bold]  {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    stats_parts = [
        f"[bold]Agents:[/bold] {mb.agents_done()}/{mb.agents_total()}",
        elapsed_str,
    ]
    if llm is not None:
        s = llm.get_stats()
        tok_in  = s.get("tokens_in",  0)
        tok_out = s.get("tokens_out", 0)
        calls   = s.get("calls",      0)
        def _fmt_k(n: int) -> str:
            return f"{n/1000:.1f}k" if n >= 1000 else str(n)
        stats_parts += [
            f"[bold]LLM calls:[/bold] {calls}",
            f"[bold]Tokens:[/bold] [green]{_fmt_k(tok_in)}[/green] in / [cyan]{_fmt_k(tok_out)}[/cyan] out",
        ]
    ft = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    ft.add_column("s", justify="center")
    ft.add_row(Text.from_markup("  [grey50]|[/grey50]  ".join(stats_parts)))
    layout["footer"].update(Panel(ft, border_style="grey50"))


# ─── Pipeline progress callback ───────────────────────────────────────────────

def build_callback(
    mb:         MessageBuffer,
    layout:     Layout,
    ticker:     str,
    exchange:   str,
    date_str:   str,
    provider:   str,
    start_time: float,
    llm=None,
):
    def _refresh():
        update_display(layout, mb, ticker, exchange, date_str, provider, start_time, llm)

    def cb(event: str, **kw):
        if event == "stage_start":
            stage = kw["stage"]
            for agent in _STAGE_AGENTS.get(stage, []):
                if mb.agent_status.get(agent) == "pending":
                    mb.update_status(agent, "in_progress")
            mb.add_message("System", f"{stage} — started")
            _refresh()

        elif event == "stage_done":
            stage = kw["stage"]
            for agent in _STAGE_AGENTS.get(stage, []):
                if mb.agent_status.get(agent) != "error":
                    mb.update_status(agent, "completed")
            mb.add_message("System", f"{stage} — complete")
            _refresh()

        elif event == "analyst_done":
            name       = kw["name"]
            signal     = kw.get("signal")
            conviction = kw.get("conviction")
            error      = kw.get("error", False)
            agent      = f"{name} Analyst"

            if error:
                mb.update_status(agent, "error")
                mb.add_message("Error", f"{agent} failed")
            else:
                mb.update_status(agent, "completed")
                color = _SIGNAL_COLOR.get(signal or "", "white")
                mb.analyst_detail[agent] = f"[{color}]{signal}[/{color}]  conv {conviction}/10"
                mb.add_message("Agent", f"{agent}: {signal} (conviction {conviction}/10)")
                mb.current_report = (
                    f"[bold]{agent}[/bold]\n"
                    f"Signal    [{color}]{signal}[/{color}]\n"
                    f"Conviction  {conviction} / 10"
                )

            _refresh()

        elif event == "cache_hit":
            for agent in mb.agent_status:
                mb.update_status(agent, "completed")
            mb.add_message("System", "Cache hit — loaded from disk, skipping pipeline")
            _refresh()

    return cb


# ─── ASCII art banner ─────────────────────────────────────────────────────────

def _print_banner() -> None:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("DALAL AGENTS", font="doom")
    except Exception:
        art = "DALAL AGENTS"

    colors = ["bright_green", "green", "bright_green", "green", "bright_green", "green"]
    t = Text()
    for i, line in enumerate(art.splitlines()):
        t.append(line + "\n", style=colors[i % len(colors)])

    console.print()
    console.print(Align.center(t))
    console.print(Align.center(
        Text("Multi-Agent LLM Trading System for Indian Markets", style="bold dim cyan")
    ))
    console.print(Align.center(
        Text("NSE  ·  BSE  ·  Powered by Large Language Models", style="dim")
    ))
    console.print()


# ─── Interactive setup wizard ─────────────────────────────────────────────────

def _qbox(title: str, body: str) -> Panel:
    return Panel(
        f"[bold]{title}[/bold]\n[dim]{body}[/dim]",
        border_style="blue",
        padding=(1, 2),
    )


def get_user_selections() -> dict:
    _print_banner()

    # Step 1 — Ticker
    console.print(_qbox(
        "Step 1: Ticker Symbol",
        "Enter a NSE/BSE ticker (e.g. RELIANCE, TCS, INFY, HDFCBANK)",
    ))
    ticker_raw = questionary.text(
        "Ticker symbol:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a ticker symbol.",
        style=_QS_GREEN,
    ).ask()
    if ticker_raw is None:
        console.print("\n[red]Cancelled.[/red]"); sys.exit(0)
    ticker = ticker_raw.strip().upper()

    # Step 2 — Analysis date
    today = datetime.date.today().isoformat()
    console.print(_qbox(
        "Step 2: Analysis Date",
        "Date of analysis in YYYY-MM-DD format",
    ))

    def _valid_date(s: str) -> bool | str:
        import re
        s = s.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return "Use YYYY-MM-DD format (e.g. 2025-01-15)"
        try:
            if datetime.date.fromisoformat(s) > datetime.date.today():
                return "Analysis date cannot be in the future"
        except ValueError:
            return "Invalid date"
        return True

    raw_date = questionary.text(
        f"Analysis date [{today}]:",
        default=today,
        validate=_valid_date,
        style=_QS_GREEN,
    ).ask()
    if raw_date is None:
        console.print("\n[red]Cancelled.[/red]"); sys.exit(0)
    analysis_date = raw_date.strip() or today

    # Step 3 — Exchange
    console.print(_qbox("Step 3: Exchange", "Select the stock exchange"))
    exchange = questionary.select(
        "Exchange:",
        choices=[
            questionary.Choice("NSE — National Stock Exchange (default)", value="NSE"),
            questionary.Choice("BSE — Bombay Stock Exchange",              value="BSE"),
        ],
        style=_QS_YELLOW,
    ).ask()
    if exchange is None:
        console.print("\n[red]Cancelled.[/red]"); sys.exit(0)

    # Step 4 — LLM provider
    console.print(_qbox("Step 4: LLM Provider", "Select the AI provider for all agents"))
    provider = questionary.select(
        "LLM provider:",
        choices=[
            questionary.Choice("Claude      (Anthropic) — recommended",  value="claude"),
            questionary.Choice("Gemini      (Google)",                    value="gemini"),
            questionary.Choice("OpenAI      (GPT-4o)",                    value="openai"),
            questionary.Choice("Grok        (xAI) — fast & cheap",        value="grok"),
            questionary.Choice("Ollama      (local, free)",               value="ollama"),
            questionary.Choice("OpenRouter  (multi-provider gateway)",    value="openrouter"),
        ],
        style=_QS_CYAN,
    ).ask()
    if provider is None:
        console.print("\n[red]Cancelled.[/red]"); sys.exit(0)

    # Step 5 — Debate rounds
    console.print(_qbox(
        "Step 5: Research Debate Rounds",
        "How many Bull vs Bear debate rounds should the research team run?\n"
        "More rounds = deeper analysis but more LLM calls and cost.",
    ))
    debate_rounds_str = questionary.select(
        "Debate rounds:",
        choices=[
            questionary.Choice("1  — quick (1 Bull + 1 Bear)",            value="1"),
            questionary.Choice("2  — balanced (default)",                  value="2"),
            questionary.Choice("3  — thorough",                            value="3"),
        ],
        default="2",
        style=_QS_YELLOW,
    ).ask()
    debate_rounds = int(debate_rounds_str) if debate_rounds_str else 2

    # Step 6 — Cache
    console.print(_qbox(
        "Step 6: Cache",
        "Skip the pipeline and return a cached result if one already exists for this ticker/date?",
    ))
    use_cache = questionary.confirm("Use cached result if available?", default=True).ask()
    if use_cache is None:
        use_cache = True

    console.print(
        f"\n[green]✓[/green] Ticker: [bold cyan]{ticker}[/bold cyan]  "
        f"Date: [bold yellow]{analysis_date}[/bold yellow]  "
        f"Exchange: [bold]{exchange}[/bold]  "
        f"Provider: [bold magenta]{provider.upper()}[/bold magenta]  "
        f"Debate rounds: [bold]{debate_rounds}[/bold]\n"
    )
    return {
        "ticker":        ticker,
        "analysis_date": analysis_date,
        "exchange":      exchange,
        "provider":      provider,
        "debate_rounds": debate_rounds,
        "use_cache":     use_cache,
    }


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _make_llm(provider: str):
    if provider == "gemini":
        if not GEMINI_API_KEY:
            console.print("[red]ERROR:[/red] GEMINI_API_KEY not set in .env"); sys.exit(1)
        return GeminiClient(model=DEFAULT_GEMINI_MODEL)
    if provider == "openai":
        if not OPENAI_API_KEY:
            console.print("[red]ERROR:[/red] OPENAI_API_KEY not set in .env"); sys.exit(1)
        return OpenAIClient(model=DEFAULT_OPENAI_MODEL)
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            console.print("[red]ERROR:[/red] OPENROUTER_API_KEY not set in .env"); sys.exit(1)
        return OpenRouterClient(model=DEFAULT_OPENROUTER_MODEL)
    if provider == "grok":
        if not GROK_API_KEY:
            console.print("[red]ERROR:[/red] GROK_API_KEY not set in .env"); sys.exit(1)
        return GrokClient(model=DEFAULT_GROK_MODEL)
    if provider == "ollama":
        console.print("[dim]Using Ollama local inference — make sure `ollama serve` is running.[/dim]")
        return OllamaClient(model=DEFAULT_OLLAMA_MODEL)
    if not ANTHROPIC_API_KEY:
        console.print("[red]ERROR:[/red] ANTHROPIC_API_KEY not set in .env"); sys.exit(1)
    return AnthropicClient(model=DEFAULT_MODEL)


# ─── Post-analysis display ────────────────────────────────────────────────────

def _fmt_inr(val) -> str:
    return "—" if val is None else f"₹{float(val):,.2f}"


def display_final_decision(fd, exchange: str, llm=None) -> None:
    console.print()
    console.print(Rule("[bold bright_blue]FINAL DECISION[/bold bright_blue]"))

    action_val = fd.action.value if hasattr(fd.action, "value") else str(fd.action)
    color  = _SIGNAL_COLOR.get(action_val, "white")
    action = f"[{color}]{action_val}[/{color}]"

    tbl = Table(box=box.ROUNDED, show_header=False, border_style="bright_blue", expand=False)
    tbl.add_column("field", style="dim",        min_width=20)
    tbl.add_column("value", style="bold white",  min_width=32)

    tbl.add_row("Ticker",        f"[bold cyan]{fd.ticker}[/bold cyan] ({exchange})")
    tbl.add_row("Date",          str(fd.as_of_date))
    tbl.add_row("Action",        action)
    tbl.add_row("Position size", f"{fd.position_size_pct}%")
    tbl.add_row("Entry price",   _fmt_inr(fd.entry_price))
    tbl.add_row("Target price",  _fmt_inr(fd.target_price))
    tbl.add_row("Stop-loss",     _fmt_inr(fd.stop_loss))
    tbl.add_row("Pipeline time", f"{fd.pipeline_duration_seconds}s")
    if llm is not None:
        s = llm.get_stats()
        def _fmt_k(n: int) -> str:
            return f"{n/1000:.1f}k" if n >= 1000 else str(n)
        tbl.add_row("LLM calls",    str(s["calls"]))
        tbl.add_row("Tokens in",    _fmt_k(s["tokens_in"]))
        tbl.add_row("Tokens out",   _fmt_k(s["tokens_out"]))
    console.print(tbl)

    if fd.rationale:
        console.print("\n[bold]Rationale[/bold]")
        for line in fd.rationale.split(". "):
            if line.strip():
                console.print(f"  {line.strip()}.")

    if fd.dissenting_view:
        console.print("\n[bold yellow]Dissenting view[/bold yellow]")
        console.print(f"  {fd.dissenting_view}")
    console.print()


def display_complete_report(state) -> None:
    """Print all analyst reports sequentially after the Live context closes."""
    console.print(Rule("[bold green]Complete Analysis Report[/bold green]"))

    if state.technical_report:
        r = state.technical_report
        console.print(Panel(
            Markdown(f"**Signal**: {r.signal}  |  **Conviction**: {r.conviction}/10\n\n{r.summary or ''}"),
            title="Technical Analyst", border_style="blue", padding=(1, 2),
        ))
    if state.sentiment_report:
        r = state.sentiment_report
        console.print(Panel(
            Markdown(f"**Signal**: {r.signal}  |  **Conviction**: {r.conviction}/10\n\n{r.summary or ''}"),
            title="Sentiment Analyst", border_style="blue", padding=(1, 2),
        ))
    if state.news_report:
        r = state.news_report
        console.print(Panel(
            Markdown(f"**Signal**: {r.signal}  |  **Conviction**: {r.conviction}/10\n\n{r.summary or ''}"),
            title="News Analyst", border_style="blue", padding=(1, 2),
        ))
    if state.fundamentals_report:
        r = state.fundamentals_report
        console.print(Panel(
            Markdown(f"**Signal**: {r.signal}  |  **Conviction**: {r.conviction}/10\n\n{r.summary or ''}"),
            title="Fundamentals Analyst", border_style="blue", padding=(1, 2),
        ))
    if state.research_debate:
        rd = state.research_debate
        console.print(Panel(
            Markdown(f"**Winner**: {rd.winning_stance}  |  **Signal**: {rd.consensus_signal}\n\n{rd.facilitator_verdict or ''}"),
            title="Research Debate", border_style="magenta", padding=(1, 2),
        ))
    if state.trade_proposal:
        tp = state.trade_proposal
        console.print(Panel(
            Markdown(
                f"**Action**: {tp.action}  |  **Position size**: {tp.position_size_pct}%  |  "
                f"**Holding period**: {tp.holding_period}\n\n"
                f"**Entry**: {tp.entry_price}  **Target**: {tp.target_price}  "
                f"**Stop**: {tp.stop_loss}  **R:R**: {tp.risk_reward_ratio}\n\n"
                f"{tp.rationale}"
            ),
            title="Trader", border_style="yellow", padding=(1, 2),
        ))
    if state.risk_debate:
        rd2 = state.risk_debate
        console.print(Panel(
            Markdown(f"**Winner**: {rd2.winning_stance}  |  **Signal**: {rd2.consensus_signal}\n\n{rd2.facilitator_verdict or ''}"),
            title="Risk Debate", border_style="magenta", padding=(1, 2),
        ))
    if state.risk_assessment:
        ra = state.risk_assessment
        console.print(Panel(
            Markdown(
                f"**Approved action**: {ra.approved_action}  |  "
                f"**Adjusted size**: {ra.adjusted_position_size_pct}%  |  "
                f"**Risk level**: {ra.risk_level}\n\n{ra.rationale}"
            ),
            title="Risk Assessment", border_style="red", padding=(1, 2),
        ))


from cli.mock import fake_pipeline as _fake_pipeline

# ─── Async pipeline runner ────────────────────────────────────────────────────

async def _run_analysis(sel: dict, *, mock: bool = False) -> None:
    ticker        = sel["ticker"]
    date_str      = sel["analysis_date"]
    exchange      = Exchange(sel["exchange"])
    provider      = sel["provider"]
    use_cache     = sel["use_cache"]
    debate_rounds = int(sel.get("debate_rounds", 2))
    start_time = time.time()

    reddit_creds = None
    if REDDIT_CLIENT_ID and not mock:
        reddit_creds = {
            "client_id":     REDDIT_CLIENT_ID,
            "client_secret": REDDIT_SECRET,
            "user_agent":    REDDIT_USER_AGENT or "DalalAgents/1.0",
        }

    llm = None if mock else _make_llm(provider)

    mb     = MessageBuffer()
    mb.init()
    layout = create_layout()

    cb = build_callback(mb, layout, ticker, sel["exchange"], date_str, provider, start_time, llm)

    # Initial render before pipeline starts
    update_display(layout, mb, ticker, sel["exchange"], date_str, provider, start_time, llm)

    with Live(layout, refresh_per_second=4, console=console):
        mb.add_message("System", f"Starting: {ticker} ({sel['exchange']})  {date_str}")
        mb.add_message("System", f"Provider: {'MOCK' if mock else provider.upper()}")
        update_display(layout, mb, ticker, sel["exchange"], date_str, provider, start_time)

        if mock:
            state = await _fake_pipeline(ticker, date_str, sel["exchange"], cb)
        else:
            state = await run_pipeline(
                ticker=ticker,
                analysis_date=datetime.date.fromisoformat(date_str),
                llm=llm,
                exchange=exchange,
                newsapi_key=NEWSAPI_KEY or "",
                reddit_creds=reddit_creds,
                skip_if_cached=use_cache,
                resume_from_checkpoint=use_cache,
                debate_rounds=debate_rounds,
                progress_callback=cb,
            )

        # Mark everything completed
        for agent in mb.agent_status:
            if mb.agent_status[agent] != "error":
                mb.update_status(agent, "completed")
        mb.add_message("System", "Analysis complete!")
        update_display(layout, mb, ticker, sel["exchange"], date_str, provider, start_time, llm)

    # Outside Live context for clean post-analysis interaction
    elapsed = time.time() - start_time
    llm_summary = ""
    if llm is not None:
        s = llm.get_stats()
        def _fmt_k(n: int) -> str:
            return f"{n/1000:.1f}k" if n >= 1000 else str(n)
        llm_summary = (
            f"  [dim]LLM calls: [bold]{s['calls']}[/bold]  "
            f"Tokens in: [green]{_fmt_k(s['tokens_in'])}[/green]  "
            f"Tokens out: [cyan]{_fmt_k(s['tokens_out'])}[/cyan][/dim]"
        )
    console.print(
        f"\n[bold cyan]Analysis Complete![/bold cyan]  "
        f"[dim]Total time: {elapsed:.1f}s[/dim]"
        + llm_summary + "\n"
    )

    if state is None or state.final_decision is None:
        console.print("[red]ERROR:[/red] Pipeline produced no final decision."); sys.exit(1)

    display_final_decision(state.final_decision, sel["exchange"], llm)

    # Offer to view full report
    show_report = typer.prompt("Display full analyst reports? [Y/n]", default="Y").strip().upper()
    if show_report in ("Y", "YES", ""):
        display_complete_report(state)

    # Offer to save state as JSON
    save_choice = typer.prompt("Save full state to disk? [Y/n]", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path.cwd() / "reports" / f"{ticker}_{date_str}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]✓ Saved:[/green] {out_path.resolve()}")


# ─── Typer entry point ────────────────────────────────────────────────────────

@app.command()
def analyze():
    """Run a full DalalAgents analysis interactively."""
    try:
        selections = get_user_selections()
        asyncio.run(_run_analysis(selections))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


@app.command()
def mock(
    ticker:   str = typer.Option("RELIANCE", help="Ticker symbol"),
    date:     str = typer.Option("", help="Analysis date YYYY-MM-DD (default: today)"),
    exchange: str = typer.Option("NSE", help="Exchange: NSE or BSE"),
):
    """Test the CLI layout without any LLM calls (uses simulated pipeline)."""
    date_str = date or datetime.date.today().isoformat()
    sel = {
        "ticker":        ticker,
        "analysis_date": date_str,
        "exchange":      exchange,
        "provider":      "mock",
        "use_cache":     False,
    }
    try:
        asyncio.run(_run_analysis(sel, mock=True))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    app()
