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
        "ROCE": "roce",
        "ROE": "roe",
        "Promoter Holding": "promoter_holding_pct",
        "Pledged percentage": "promoter_pledge_pct",
        "FII": "fii_holding_pct",
        "DII": "dii_holding_pct",
        "Market Cap": "market_cap_cr",
        "Dividend Yield": "dividend_yield",
    }
    price_labels = {
        "Current Price": "current_price",
        "Book Value": "book_value",
    }

    result: dict = {
        "source": "screener.in",
        "ticker": ticker,
        "as_of_date": str(as_of_date),
    }
    prices: dict = {}

    for li in soup.find_all("li"):
        name_span = li.find("span", class_="name")
        if not name_span:
            continue
        label = name_span.get_text(strip=True)
        if label not in label_map and label not in price_labels:
            continue
        val_span = li.find("span", class_=["number", "value"])
        if not val_span:
            continue
        raw = (
            val_span.get_text(strip=True)
            .replace(",", "")
            .replace("%", "")
            .replace("₹", "")
            .replace("Cr.", "")
            .strip()
        )
        try:
            value = float(raw)
        except ValueError:
            value = None
        if label in label_map:
            result[label_map[label]] = value
        else:
            prices[price_labels[label]] = value

    result["pb_ratio"] = (
        round(prices["current_price"] / prices["book_value"], 2)
        if prices.get("current_price") and prices.get("book_value")
        else None
    )

    shareholding_map = {
        "Promoters": "promoter_holding_pct",
        "FIIs": "fii_holding_pct",
        "DIIs": "dii_holding_pct",
    }
    quarterly_shp = soup.find("div", id="quarterly-shp")
    if quarterly_shp:
        for row in quarterly_shp.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            row_label = cells[0].get_text(strip=True).rstrip("+").strip()
            if row_label not in shareholding_map:
                continue
            raw = cells[-1].get_text(strip=True).replace("%", "").replace(",", "").strip()
            try:
                result[shareholding_map[row_label]] = float(raw)
            except ValueError:
                result[shareholding_map[row_label]] = None

    balance_sheet_map = {
        "Equity Capital": "equity_capital",
        "Reserves": "reserves",
        "Borrowings": "borrowings",
    }
    balance_sheet = soup.find("section", id="balance-sheet")
    balance_sheet_vals: dict = {}
    if balance_sheet:
        for row in balance_sheet.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            row_label = cells[0].get_text(strip=True).rstrip("+").strip()
            if row_label not in balance_sheet_map:
                continue
            raw = cells[-1].get_text(strip=True).replace(",", "").strip()
            try:
                balance_sheet_vals[balance_sheet_map[row_label]] = float(raw)
            except ValueError:
                pass

    equity = balance_sheet_vals.get("equity_capital", 0) + balance_sheet_vals.get("reserves", 0)
    result["debt_to_equity"] = (
        round(balance_sheet_vals["borrowings"] / equity, 3)
        if "borrowings" in balance_sheet_vals and equity
        else None
    )

    for key in [*label_map.values(), "pb_ratio", "debt_to_equity"]:
        result.setdefault(key, None)

    return result
