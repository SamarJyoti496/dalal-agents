"""
pipeline/calendar.py — NSE trading calendar derived from Nifty 50 OHLCV.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


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
    return [idx.date() for idx in df.index if start_date <= idx.date() <= end_date]
