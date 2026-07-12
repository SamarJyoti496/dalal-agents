from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from dalal_agents.tools.guards import _check_lookahead, _safe_float
from dalal_agents.tools.market import get_ohlcv


def get_technical_indicators(
    ticker_symbol: str,
    as_of_date: date,
    lookback_days: int = 100,
) -> dict:
    _check_lookahead(as_of_date)
    import pandas_ta as ta

    df = get_ohlcv(ticker_symbol, as_of_date, lookback_days)
    close = round(float(df["Close"].iloc[-1]), 2)

    # RSI
    rsi_series = ta.rsi(df["Close"], length=14)
    rsi_14 = _safe_float(rsi_series, -1)

    # MACD — detect crossover by comparing yesterday and today
    macd_df = ta.macd(df["Close"])
    macd_signal_str = "neutral"
    if macd_df is not None and not macd_df.empty and len(macd_df) >= 2:
        macd_col = next((c for c in macd_df.columns if c.startswith("MACD_")), None)
        sig_col = next((c for c in macd_df.columns if c.startswith("MACDs_")), None)
        if macd_col and sig_col:
            m_today = macd_df[macd_col].iloc[-1]
            s_today = macd_df[sig_col].iloc[-1]
            m_prev = macd_df[macd_col].iloc[-2]
            s_prev = macd_df[sig_col].iloc[-2]
            if pd.notna(m_today) and pd.notna(s_today):
                if m_prev <= s_prev and m_today > s_today:
                    macd_signal_str = "bullish_crossover"
                elif m_prev >= s_prev and m_today < s_today:
                    macd_signal_str = "bearish_crossover"

    # ADX
    adx_df = ta.adx(df["High"], df["Low"], df["Close"])
    adx = None
    if adx_df is not None:
        adx_col = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
        if adx_col:
            adx = _safe_float(adx_df[adx_col], -1)

    # Bollinger Bands
    bb_df = ta.bbands(df["Close"], length=20)
    bb_position = "inside"
    if bb_df is not None:
        bbu_col = next((c for c in bb_df.columns if c.startswith("BBU_")), None)
        bbl_col = next((c for c in bb_df.columns if c.startswith("BBL_")), None)
        if bbu_col and bbl_col:
            bbu = bb_df[bbu_col].iloc[-1]
            bbl = bb_df[bbl_col].iloc[-1]
            if pd.notna(bbu) and pd.notna(bbl):
                if close > float(bbu):
                    bb_position = "above_upper"
                elif close < float(bbl):
                    bb_position = "below_lower"

    # EMA 20 and 50
    ema_20 = _safe_float(ta.ema(df["Close"], length=20), -1)
    ema_50 = _safe_float(ta.ema(df["Close"], length=50), -1)
    ema_20_vs_50 = "unknown"
    if ema_20 is not None and ema_50 is not None:
        ema_20_vs_50 = "golden_cross" if ema_20 > ema_50 else "death_cross"

    # ATR as percentage of close price
    atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    atr_pct = None
    if atr_series is not None:
        atr_val = atr_series.iloc[-1]
        if pd.notna(atr_val) and close > 0:
            atr_pct = round(float(atr_val) / close * 100, 2)

    # VWAP - cumulative over the lookback window
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap_raw = (typical * df["Volume"]).sum() / df["Volume"].sum()
    vwap = round(float(vwap_raw), 2) if pd.notna(vwap_raw) else None
    vwap_position = "above" if (vwap is not None and close > vwap) else "below"

    return {
        "close": close,
        "rsi_14": rsi_14,
        "macd_signal": macd_signal_str,
        "adx": adx,
        "bb_position": bb_position,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_20_vs_50": ema_20_vs_50,
        "atr_pct": atr_pct,
        "vwap": vwap,
        "vwap_position": vwap_position,
    }
