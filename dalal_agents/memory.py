"""
dalal_agents/memory.py — Cross-run decision memory for DalalAgents.

After every completed pipeline run the final decision is appended to
MEMORY_PATH (default: ./dalal_memory.md in the project root).

The FundManagerAgent loads the last N decisions for the same ticker
before making its call so it can factor in "what we decided last time
and whether the thesis played out".  This is the same pattern used by
TauricResearch/TradingAgents — institutional memory across runs.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dalal_agents.models import TradingState

MEMORY_PATH: Path = Path(__file__).resolve().parent.parent / "dalal_memory.md"
_HEADER = "# DalalAgents Decision Memory\n\nAuto-generated — do not edit manually.\n\n"


# =============================================================================
# Write
# =============================================================================

def save_decision(state: "TradingState", memory_path: Path = MEMORY_PATH) -> None:
    """Append the final decision from a completed run to the memory file."""
    if state.final_decision is None:
        return
    fd = state.final_decision
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    entry_lines = [
        f"## {state.ticker} | {fd.as_of_date} | recorded {ts}",
        f"- **Exchange:** {state.exchange.value}",
        f"- **Action:** {fd.action.value}  |  **Size:** {fd.position_size_pct}%",
        f"- **Entry:** ₹{fd.entry_price}  **Target:** ₹{fd.target_price}  "
        f"**Stop:** ₹{fd.stop_loss}",
        f"- **Pipeline time:** {fd.pipeline_duration_seconds}s",
        f"- **Rationale:** {fd.rationale}",
    ]
    if fd.dissenting_view:
        entry_lines.append(f"- **Dissent:** {fd.dissenting_view}")
    entry_lines.append("")   # blank line between entries

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if not memory_path.exists():
        memory_path.write_text(_HEADER, encoding="utf-8")

    with memory_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(entry_lines))


# =============================================================================
# Read
# =============================================================================

def load_recent_decisions(
    ticker: str,
    n: int = 5,
    memory_path: Path = MEMORY_PATH,
) -> str:
    """
    Return a formatted string of the last N decisions for ticker, ready to
    inject verbatim into an LLM prompt.  Returns "" if no history exists.
    """
    if not memory_path.exists():
        return ""

    text = memory_path.read_text(encoding="utf-8")
    # Each entry starts with "## {TICKER} |"
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith(f"## {ticker} |"):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    if not blocks:
        return ""

    recent = blocks[-n:]
    joined = "\n\n".join(recent)
    return (
        f"=== PRIOR DECISIONS FOR {ticker} "
        f"(last {len(recent)} of {len(blocks)} recorded) ===\n"
        f"{joined}\n"
        "=== END PRIOR DECISIONS ==="
    )
