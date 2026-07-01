from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd


def _check_lookahead(as_of_date: date) -> None:
    today = date.today()
    if as_of_date > today:
        raise ValueError(
            f"Look-ahead bias prevented: requested data for {as_of_date} "
            f"but today is {today}."
        )


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns yfinance sometimes returns for single tickers."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _safe_float(series: pd.Series, idx: int) -> Optional[float]:
    try:
        val = series.iloc[idx]
        return round(float(val), 2) if pd.notna(val) else None
    except Exception:
        return None
