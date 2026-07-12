from __future__ import annotations

from datetime import date

import requests
from bs4 import BeautifulSoup

from dalal_agents.tools.guards import _check_lookahead


def get_screener_fundamentals(ticker: str, as_of_date: date) -> dict:
    _check_lookahead(as_of_date)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    url = f"https://www.screener.in/company/{ticker}/consolidated/"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        return {
            "error": str(exc),
            "ticker": ticker,
            "source": "screener.in",
            "as_of_date": str(as_of_date),
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    label_map = {
        "Stock P/E": "pe_ratio",
        "Price to Book": "pb_ratio",
        "ROCE": "roce",
        "ROE": "roe",
        "Debt to equity": "debt_to_equity",
        "Promoter Holding": "promoter_holding_pct",
        "Pledged percentage": "promoter_pledge_pct",
        "FII": "fii_holding_pct",
        "DII": "dii_holding_pct",
        "Market Cap": "market_cap_cr",
        "Dividend Yield": "dividend_yield",
        "Current ratio": "current_ratio",
        "Interest Coverage": "interest_coverage",
    }

    result: dict = {
        "source": "screener.in",
        "ticker": ticker,
        "as_of_date": str(as_of_date),
    }

    for li in soup.find_all("li"):
        name_span = li.find("span", class_="name")
        if not name_span:
            continue
        label = name_span.get_text(strip=True)
        if label not in label_map:
            continue
        val_span = li.find("span", class_=["number", "value"])
        if val_span:
            raw = val_span.get_text(strip=True).replace(",", "").replace("%", "").strip()
            try:
                result[label_map[label]] = float(raw)
            except ValueError:
                result[label_map[label]] = None

    for key in label_map.values():
        result.setdefault(key, None)

    return result
