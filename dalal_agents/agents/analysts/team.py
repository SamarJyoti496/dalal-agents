from __future__ import annotations

import asyncio
import logging
from typing import Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from dalal_agents.display import SIGNAL_STYLE as _SIGNAL_STYLE
from dalal_agents.models import TradingState
from dalal_agents.agents.analysts.fundamentals import FundamentalsAnalystAgent
from dalal_agents.agents.analysts.news import NewsAnalystAgent
from dalal_agents.agents.analysts.sentiment import SentimentAnalystAgent
from dalal_agents.agents.analysts.technical import TechnicalAnalystAgent

_console = Console()
logger = logging.getLogger("dalal_agents.agents")


class AnalystTeam:
    """
    Runs all four specialist analysts concurrently and writes their reports
    directly onto the TradingState blackboard.
    """

    def __init__(
        self,
        llm,
        newsapi_key: str = "",
        reddit_creds: Optional[dict] = None,
        on_agent_done=None,
    ):
        self.technical = TechnicalAnalystAgent(llm=llm)
        self.sentiment = SentimentAnalystAgent(
            llm=llm, newsapi_key=newsapi_key, reddit_creds=reddit_creds
        )
        self.news = NewsAnalystAgent(llm=llm, newsapi_key=newsapi_key)
        self.fundamentals = FundamentalsAnalystAgent(llm=llm)
        self._on_agent_done = on_agent_done

    async def run(self, state: TradingState) -> TradingState:
        agents = [
            ("Technical", "technical_report", self.technical),
            ("Sentiment", "sentiment_report", self.sentiment),
            ("News", "news_report", self.news),
            ("Fundamentals", "fundamentals_report", self.fundamentals),
        ]
        on_done = self._on_agent_done

        _status: dict[str, tuple] = {n: ("pending", None, None) for n, _, _ in agents}
        _errors: dict[str, str] = {}

        def _make_table() -> Table:
            tbl = Table(
                box=box.ROUNDED,
                border_style="cyan",
                header_style="bold magenta",
                expand=True,
                show_footer=False,
                padding=(0, 1),
            )
            tbl.add_column("Agent", style="bold white", min_width=20)
            tbl.add_column("Signal", min_width=12)
            tbl.add_column("Conviction", justify="center", min_width=10)
            tbl.add_column("Status", min_width=14)

            for name, _, _ in agents:
                status, signal, conviction = _status[name]
                if status == "pending":
                    tbl.add_row(
                        f"{name}Analyst",
                        Text("─", style="dim"),
                        Text("─", style="dim"),
                        Text("○ pending", style="dim"),
                    )
                elif status == "running":
                    tbl.add_row(
                        f"{name}Analyst",
                        Text("─", style="dim"),
                        Text("─", style="dim"),
                        Text("⠙ running…", style="bold cyan"),
                    )
                elif status == "done":
                    style = _SIGNAL_STYLE.get(signal or "", "white")
                    tbl.add_row(
                        f"{name}Analyst",
                        Text(signal or "─", style=style),
                        Text(f"{conviction}/10" if conviction else "─"),
                        Text("✓ done", style="bold green"),
                    )
                else:
                    tbl.add_row(
                        f"{name}Analyst",
                        Text("─", style="dim"),
                        Text("─", style="dim"),
                        Text("✗ failed", style="bold red"),
                    )
            return tbl

        async def _run_one(name: str, field: str, agent, live=None) -> None:
            if live is not None:
                _status[name] = ("running", None, None)
                live.update(_make_table())
            try:
                result = await agent.run(state)
                setattr(state, field, result)
                sig = str(result.signal).replace("Signal.", "")
                if live is not None:
                    _status[name] = ("done", sig, result.conviction)
                    live.update(_make_table())
                if on_done:
                    on_done(name, sig, result.conviction, False)
            except Exception as exc:
                if live is not None:
                    _status[name] = ("error", None, None)
                    live.update(_make_table())
                logger.exception("%sAnalyst failed", name)
                _errors[name] = f"{type(exc).__name__}: {exc}"
                if on_done:
                    on_done(name, None, None, True)

        if on_done:
            await asyncio.gather(*[_run_one(n, f, a) for n, f, a in agents])
        else:
            with Live(_make_table(), console=_console, refresh_per_second=8) as live:
                await asyncio.gather(*[_run_one(n, f, a, live) for n, f, a in agents])

        for name, err in _errors.items():
            _console.print(f"  [bold red]✗[/bold red] {name}Analyst: [red]{err}[/red]")

        return state
