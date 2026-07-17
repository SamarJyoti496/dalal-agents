"""Shared Rich-console styling constants."""

# Signal -> Rich style string, used to color STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
# wherever a signal is rendered to the terminal.
SIGNAL_STYLE: dict[str, str] = {
    "STRONG_BUY": "bold green",
    "BUY": "green",
    "HOLD": "yellow",
    "SELL": "red",
    "STRONG_SELL": "bold red",
}
