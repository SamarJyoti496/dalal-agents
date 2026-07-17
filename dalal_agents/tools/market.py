from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from dalal_agents.tools.guards import _check_lookahead, _flatten_columns, _strip_tz
from dalal_agents.tools.sector_index import SECTOR_INDEX, SECTOR_NAMES


def _nse_get(path: str, params: dict | None = None) -> dict | list:
    """
    Make a GET request to the NSE India REST API, seeding the required
    browser-like cookies first.  path must start with "/api/".
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
    )
    # Seed cookies by hitting the homepage first
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(0.5)
    resp = session.get(
        f"https://www.nseindia.com{path}",
        params=params or {},
        headers={"Referer": "https://www.nseindia.com/"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def get_ohlcv(
    ticker_symbol: str,
    as_of_date: date,
    lookback_days: int = 60,
) -> pd.DataFrame:
    _check_lookahead(as_of_date)

    start = as_of_date - timedelta(days=lookback_days)
    end = as_of_date + timedelta(days=1)  # yfinance end is exclusive

    df = yf.download(
        ticker_symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    df = _flatten_columns(df)
    df = _strip_tz(df)

    # Second filter — yfinance sometimes leaks one row past the requested end
    df = df[df.index <= pd.Timestamp(as_of_date)]

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]

    # Drop trailing rows with no Close — yfinance sometimes publishes a row for the
    # most recent session (Volume already ticking) before it has finalized the OHLC.
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker_symbol}' up to {as_of_date}. "
            "Verify the ticker ends with .NS (NSE) or .BO (BSE)."
        )
    return df


def get_india_vix(as_of_date: date) -> Optional[float]:
    _check_lookahead(as_of_date)
    try:
        df = yf.download(
            "^INDIAVIX",
            start=as_of_date - timedelta(days=10),
            end=as_of_date + timedelta(days=1),
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        df = _flatten_columns(df)
        df = _strip_tz(df)
        df = df[df.index <= pd.Timestamp(as_of_date)]
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])
        if df.empty or "Close" not in df.columns:
            return None
        return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        return None


def get_nifty_context(as_of_date: date) -> dict:
    _check_lookahead(as_of_date)
    import pandas_ta as ta

    df = yf.download(
        "^NSEI",
        start=as_of_date - timedelta(days=120),
        end=as_of_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    df = _flatten_columns(df)
    df = _strip_tz(df)
    df = df[df.index <= pd.Timestamp(as_of_date)]
    df = df.dropna(subset=["Close"])

    close = round(float(df["Close"].iloc[-1]), 2)

    five_day_return = None
    if len(df) >= 6:
        prev = float(df["Close"].iloc[-6])
        five_day_return = round((close - prev) / prev * 100, 2)

    ema20_val = ta.ema(df["Close"], length=20).iloc[-1]
    ema50_val = ta.ema(df["Close"], length=50).iloc[-1]

    above_ema20 = bool(pd.notna(ema20_val) and close > float(ema20_val))
    above_ema50 = bool(pd.notna(ema50_val) and close > float(ema50_val))

    trend = "unknown"
    if pd.notna(ema50_val):
        trend = "bullish" if close > float(ema50_val) else "bearish"

    india_vix = get_india_vix(as_of_date)
    fear_level = "normal"
    if india_vix is not None:
        if india_vix > 30:
            fear_level = "extreme"
        elif india_vix > 20:
            fear_level = "high"

    return {
        "nifty50_close": close,
        "five_day_return": five_day_return,
        "above_ema20": above_ema20,
        "above_ema50": above_ema50,
        "trend": trend,
        "india_vix": india_vix,
        "fear_level": fear_level,
    }


def get_sector_index_context(ticker: str, as_of_date: date) -> dict:
    """
    Return the trend of the sector index most relevant to *ticker*.

    Falls back to Nifty 50 if no mapping is found.  Use this alongside
    get_nifty_context to distinguish between broad-market moves and
    sector-rotation moves.
    """
    _check_lookahead(as_of_date)
    import pandas_ta as ta

    sector_sym = SECTOR_INDEX.get(ticker.upper(), "^NSEI")
    sector_name = SECTOR_NAMES.get(sector_sym, sector_sym)

    try:
        df = yf.download(
            sector_sym,
            start=as_of_date - timedelta(days=120),
            end=as_of_date + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        df = _flatten_columns(df)
        df = _strip_tz(df)
        df = df[df.index <= pd.Timestamp(as_of_date)]
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])

        if df.empty or "Close" not in df.columns:
            return {"error": f"No data for {sector_sym}", "sector": sector_name}

        close = round(float(df["Close"].iloc[-1]), 2)
        ema50_val = ta.ema(df["Close"], length=50).iloc[-1]
        trend = "bullish" if (pd.notna(ema50_val) and close > float(ema50_val)) else "bearish"

        five_day_return = None
        if len(df) >= 6:
            prev = float(df["Close"].iloc[-6])
            five_day_return = round((close - prev) / prev * 100, 2)

        twenty_day_return = None
        if len(df) >= 21:
            prev20 = float(df["Close"].iloc[-21])
            twenty_day_return = round((close - prev20) / prev20 * 100, 2)

        return {
            "sector": sector_name,
            "sector_symbol": sector_sym,
            "close": close,
            "trend": trend,
            "five_day_return_pct": five_day_return,
            "twenty_day_return_pct": twenty_day_return,
        }
    except Exception as exc:
        return {"error": str(exc), "sector": sector_name, "sector_symbol": sector_sym}


def get_corporate_actions(ticker: str, as_of_date: date, lookforward_days: int = 45) -> list[dict]:
    """
    Fetch upcoming (and recent past) corporate actions for *ticker* from NSE:
    dividends, bonuses, splits, rights issues, board meetings.

    lookforward_days: how far into the future to scan for upcoming events.
    """
    _check_lookahead(as_of_date)

    from_date = (as_of_date - timedelta(days=10)).strftime("%d-%m-%Y")
    to_date = (as_of_date + timedelta(days=lookforward_days)).strftime("%d-%m-%Y")

    try:
        data = _nse_get(
            "/api/corporates-corporateActions",
            params={
                "index": "equities",
                "from_date": from_date,
                "to_date": to_date,
                "symbol": ticker.upper(),
            },
        )
    except Exception as exc:
        return [{"error": f"NSE corporate actions fetch failed: {exc}"}]

    if not isinstance(data, list):
        data = data.get("data", []) if isinstance(data, dict) else []

    results = []
    for row in data:
        sym = (row.get("symbol") or row.get("Symbol") or "").upper()
        if sym != ticker.upper():
            continue
        results.append(
            {
                "purpose": row.get("purpose") or row.get("subject") or "",
                "ex_date": row.get("exDate") or row.get("ex_date") or "",
                "record_date": row.get("recDate") or row.get("record_date") or "",
                "bc_start": row.get("bcStartDate") or "",
                "bc_end": row.get("bcEndDate") or "",
            }
        )

    return (
        results if results else [{"info": f"No corporate actions found for {ticker} in the window"}]
    )
