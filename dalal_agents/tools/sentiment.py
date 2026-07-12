from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from dalal_agents.tools.guards import _check_lookahead
from dalal_agents.tools.market import _nse_get


def get_fii_dii_flows(as_of_date: date, lookback_days: int = 5) -> list[dict]:
    _check_lookahead(as_of_date)

    # Attempt 1: nsefin library
    try:
        import nsefin

        raw = nsefin.get_fii_dii_data()
        if raw is not None:
            return _parse_nsefin_rows(raw, as_of_date, lookback_days)
    except Exception:
        pass

    # Attempt 2: NSE India REST API with a seeded browser session
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers={"Referer": "https://www.nseindia.com/"},
            timeout=15,
        )
        resp.raise_for_status()
        return _parse_nse_api_rows(resp.json(), as_of_date, lookback_days)
    except Exception as exc:
        return [{"error": f"All FII/DII sources failed: {exc}"}]


def _parse_nse_api_rows(raw: list, as_of_date: date, lookback_days: int) -> list[dict]:
    start = as_of_date - timedelta(days=lookback_days * 2)
    by_date: dict[date, dict] = {}

    for row in raw:
        try:
            row_date = datetime.strptime(row.get("date", ""), "%d-%b-%Y").date()
        except ValueError:
            continue
        if row_date > as_of_date or row_date < start:
            continue

        category = row.get("category", "").upper()
        net = round(float(row.get("buyValue") or 0) - float(row.get("saleValue") or 0), 2)
        entry = by_date.setdefault(
            row_date, {"date": str(row_date), "fii_net_cr": 0.0, "dii_net_cr": 0.0}
        )
        if "FII" in category:
            entry["fii_net_cr"] = round(entry["fii_net_cr"] + net, 2)
        elif "DII" in category:
            entry["dii_net_cr"] = round(entry["dii_net_cr"] + net, 2)

    for entry in by_date.values():
        entry["combined_net_cr"] = round(entry["fii_net_cr"] + entry["dii_net_cr"], 2)

    return sorted(by_date.values(), key=lambda x: x["date"], reverse=True)[:lookback_days]


def _parse_nsefin_rows(raw, as_of_date: date, lookback_days: int) -> list[dict]:
    rows = list(raw.iterrows()) if hasattr(raw, "iterrows") else [(None, r) for r in raw]
    results = []
    for _, rec in rows:
        rec = rec if isinstance(rec, dict) else rec.to_dict()
        try:
            rec_date = pd.Timestamp(rec.get("date") or rec.get("Date")).date()
        except Exception:
            continue
        if rec_date > as_of_date:
            continue
        fii = float(rec.get("FII_net") or rec.get("fii_net") or 0)
        dii = float(rec.get("DII_net") or rec.get("dii_net") or 0)
        results.append(
            {
                "date": str(rec_date),
                "fii_net_cr": round(fii, 2),
                "dii_net_cr": round(dii, 2),
                "combined_net_cr": round(fii + dii, 2),
            }
        )
    return sorted(results, key=lambda x: x["date"], reverse=True)[:lookback_days]


def get_india_stock_news(
    company_name: str,
    as_of_date: date,
    lookback_days: int = 7,
    newsapi_key: str = "",
) -> list[dict]:
    _check_lookahead(as_of_date)

    if not newsapi_key:
        return [{"error": "newsapi_key not configured. Set NEWSAPI_KEY in .env"}]

    from_date = as_of_date - timedelta(days=lookback_days)

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": company_name,
                "from": str(from_date),
                "to": str(as_of_date),
                "language": "en",
                "sortBy": "relevancy",
                "apiKey": newsapi_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return [{"error": f"NewsAPI request failed: {exc}"}]

    articles = []
    for item in data.get("articles", []):
        title = item.get("title", "")
        if "[Removed]" in title:
            continue
        articles.append(
            {
                "title": title,
                "source": item.get("source", {}).get("name", ""),
                "published_at": item.get("publishedAt", ""),
                "description": item.get("description", ""),
            }
        )
    return articles


def get_india_reddit_sentiment(
    query: str,
    as_of_date: date,
    lookback_days: int = 7,
    reddit_client_id: str = "",
    reddit_secret: str = "",
    reddit_user_agent: str = "DalalAgents/1.0",
) -> dict:
    _check_lookahead(as_of_date)

    if not reddit_client_id or not reddit_secret:
        return {"error": "reddit_not_configured. Set REDDIT_CLIENT_ID and REDDIT_SECRET in .env"}

    import praw
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    window_start = datetime.combine(
        as_of_date - timedelta(days=lookback_days), datetime.min.time()
    ).timestamp()
    window_end = datetime.combine(as_of_date, datetime.max.time()).timestamp()

    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_secret,
        user_agent=reddit_user_agent,
    )
    analyzer = SentimentIntensityAnalyzer()
    posts = []

    for sub_name in ("IndiaInvestments", "DalalStreet", "IndianStockMarket"):
        try:
            for post in reddit.subreddit(sub_name).search(query, limit=50, sort="new"):
                if window_start <= post.created_utc <= window_end:
                    text = f"{post.title} {post.selftext or ''}".strip()
                    compound = analyzer.polarity_scores(text)["compound"]
                    posts.append(
                        {
                            "title": post.title,
                            "subreddit": sub_name,
                            "score": post.score,
                            "sentiment": round(compound, 4),
                            "url": f"https://reddit.com{post.permalink}",
                        }
                    )
        except Exception:
            continue

    if not posts:
        return {"post_count": 0, "avg_sentiment": 0.0, "overall": "neutral", "posts": []}

    avg = round(sum(p["sentiment"] for p in posts) / len(posts), 4)
    overall = "neutral"
    if avg > 0.05:
        overall = "bullish"
    elif avg < -0.05:
        overall = "bearish"

    return {
        "post_count": len(posts),
        "avg_sentiment": avg,
        "overall": overall,
        "posts": sorted(posts, key=lambda x: x["score"], reverse=True)[:10],
    }


def get_nse_options_pcr(ticker: str, as_of_date: date) -> dict:
    """
    Fetch the NSE options chain for an F&O stock and compute the
    Put-Call Ratio (PCR) by open interest.

    PCR > 1.2  → heavy put buying → bearish sentiment / fear
    PCR < 0.7  → call-heavy → bullish sentiment / complacency
    PCR 0.7–1.2 → neutral

    Works only for F&O-eligible NSE stocks.  Returns an error dict
    for stocks without a listed options chain.
    """
    _check_lookahead(as_of_date)

    try:
        data = _nse_get(
            "/api/option-chain-equities",
            params={"symbol": ticker.upper()},
        )
    except Exception as exc:
        return {"error": f"Options chain fetch failed: {exc}", "ticker": ticker}

    records = data.get("records", {}).get("data", [])
    if not records:
        return {"error": "Empty options chain — stock may not be F&O eligible", "ticker": ticker}

    total_call_oi = 0
    total_put_oi = 0
    atm_call_oi = 0
    atm_put_oi = 0

    underlying = data.get("records", {}).get("underlyingValue", 0.0) or 0.0

    # Find ATM strike (closest to spot)
    strikes = sorted({r.get("strikePrice", 0) for r in records})
    atm_strike = min(strikes, key=lambda x: abs(x - underlying)) if strikes else 0

    for row in records:
        strike = row.get("strikePrice", 0)
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        call_oi = ce.get("openInterest", 0) or 0
        put_oi = pe.get("openInterest", 0) or 0
        total_call_oi += call_oi
        total_put_oi += put_oi
        if strike == atm_strike:
            atm_call_oi = call_oi
            atm_put_oi = put_oi

    pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0.0
    atm_pcr = round(atm_put_oi / atm_call_oi, 3) if atm_call_oi > 0 else 0.0

    sentiment = "neutral"
    if pcr > 1.2:
        sentiment = "bearish"
    elif pcr < 0.7:
        sentiment = "bullish"

    return {
        "ticker": ticker.upper(),
        "underlying_price": round(float(underlying), 2),
        "atm_strike": atm_strike,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pcr": pcr,
        "atm_pcr": atm_pcr,
        "options_sentiment": sentiment,
        "interpretation": (
            f"PCR={pcr:.2f}: heavy put activity → bearish/hedging"
            if pcr > 1.2
            else (
                f"PCR={pcr:.2f}: call-heavy → bullish/complacent"
                if pcr < 0.7
                else f"PCR={pcr:.2f}: balanced put/call → neutral"
            )
        ),
    }


def get_bulk_block_deals(ticker: str, as_of_date: date, lookback_days: int = 30) -> list[dict]:
    """
    Fetch NSE bulk and block deals for *ticker* over the past lookback_days.

    Bulk deals (> 0.5% of equity in a single transaction) and block deals
    (negotiated large trades on a dedicated window) reveal institutional
    conviction.  Promoter entities buying in bulk is a very bullish signal;
    large FII selling in bulk warrants caution.
    """
    _check_lookahead(as_of_date)

    from_dt = as_of_date - timedelta(days=lookback_days)
    from_str = from_dt.strftime("%d-%m-%Y")
    to_str = as_of_date.strftime("%d-%m-%Y")

    results: list[dict] = []

    # Bulk deals
    try:
        bulk = _nse_get(
            "/api/historical/bulk-deals",
            params={"from": from_str, "to": to_str},
        )
        bulk_rows = bulk if isinstance(bulk, list) else bulk.get("data", [])
        for row in bulk_rows:
            sym = (row.get("symbol") or row.get("Symbol") or "").upper()
            if sym != ticker.upper():
                continue
            results.append(
                {
                    "type": "BULK",
                    "date": row.get("date") or row.get("BD_DT_DATE") or "",
                    "client": row.get("clientName") or row.get("BD_CLIENT_NAME") or "",
                    "buy_sell": row.get("buySell") or row.get("BD_BUY_SELL") or "",
                    "quantity": row.get("quantityTraded") or row.get("BD_QTY_TRD") or 0,
                    "price": row.get("tradePrice") or row.get("BD_TP_WATP") or 0.0,
                }
            )
    except Exception:
        pass

    # Block deals
    try:
        block = _nse_get(
            "/api/historical/block-deals",
            params={"from": from_str, "to": to_str},
        )
        block_rows = block if isinstance(block, list) else block.get("data", [])
        for row in block_rows:
            sym = (row.get("symbol") or row.get("Symbol") or "").upper()
            if sym != ticker.upper():
                continue
            results.append(
                {
                    "type": "BLOCK",
                    "date": row.get("date") or row.get("BD_DT_DATE") or "",
                    "client": row.get("clientName") or row.get("BD_CLIENT_NAME") or "",
                    "buy_sell": row.get("buySell") or row.get("BD_BUY_SELL") or "",
                    "quantity": row.get("quantityTraded") or row.get("BD_QTY_TRD") or 0,
                    "price": row.get("tradePrice") or row.get("BD_TP_WATP") or 0.0,
                }
            )
    except Exception:
        pass

    if not results:
        return [
            {"info": f"No bulk/block deals found for {ticker} in the past {lookback_days} days"}
        ]

    return sorted(results, key=lambda x: x["date"], reverse=True)


def get_google_news_headlines(
    query: str,
    as_of_date: date,
    lookback_days: int = 7,
) -> list[dict]:
    """
    Fetch recent headlines from Google News RSS for *query* filtered to
    Indian business sources (Economic Times, Moneycontrol, Business Standard,
    Mint, LiveMint, etc.).

    No API key required — uses the public Google News RSS feed.
    Returns up to 20 headlines with title, source, and published date.
    """
    _check_lookahead(as_of_date)

    try:
        import feedparser
    except ImportError:
        return [{"error": "feedparser not installed. Run: pip install feedparser"}]

    cutoff = as_of_date - timedelta(days=lookback_days)

    # Use India locale + restrict to business/finance sources
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query)}"
        f"+when:{lookback_days}d"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        feed = feedparser.parse(rss_url)
    except Exception as exc:
        return [{"error": f"RSS fetch failed: {exc}"}]

    _INDIA_SOURCES = {
        "economic times",
        "moneycontrol",
        "business standard",
        "mint",
        "livemint",
        "the hindu businessline",
        "ndtv profit",
        "financial express",
        "cnbctv18",
        "zee business",
        "bloomberg quint",
        "reuters india",
        "business today",
    }

    articles = []
    for entry in feed.entries[:30]:
        source = (entry.get("source", {}) or {}).get("title", "").lower()
        title = entry.get("title", "")
        pub = entry.get("published", "")

        # Parse publish date
        pub_date = None
        if entry.get("published_parsed"):
            import calendar

            pub_date = datetime.utcfromtimestamp(calendar.timegm(entry.published_parsed)).date()

        if pub_date and pub_date < cutoff:
            continue

        # Prefer Indian sources but include all if not enough
        is_india_source = any(s in source for s in _INDIA_SOURCES)
        articles.append(
            {
                "title": title,
                "source": (
                    entry.get("source", {}).get("title", "")
                    if isinstance(entry.get("source"), dict)
                    else source
                ),
                "published": pub,
                "india_source": is_india_source,
            }
        )

    # Sort: Indian sources first, then by date
    articles.sort(key=lambda x: (not x["india_source"], x["published"]), reverse=False)
    return articles[:20]
