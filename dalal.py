#!/usr/bin/env python3
"""
DalalAgents CLI entry point.

Usage:
  python dalal.py run    TICKER [--date YYYY-MM-DD] [--exchange NSE|BSE]
  python dalal.py show   TICKER [--date YYYY-MM-DD]
  python dalal.py history TICKER
  python dalal.py list
  python dalal.py backtest TICKER --start YYYY-MM-DD --end YYYY-MM-DD
"""
from cli.commands import main

if __name__ == "__main__":
    main()
