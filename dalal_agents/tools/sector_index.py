"""
Sector index mappings for NSE tickers.
"""
from __future__ import annotations



SECTORS: list[dict] = [
    {
        "name": "Nifty Bank",
        "index_symbol": "^NSEBANK",
        "tickers": [
            "HDFCBANK",
            "ICICIBANK",
            "KOTAKBANK",
            "SBIN",
            "AXISBANK",
            "BANKBARODA",
            "INDUSINDBK",
            "FEDERALBNK",
            "IDFCFIRSTB",
            "AUBANK",
            "CSBBANK",
        ],
    },
    {
        "name": "Nifty IT",
        "index_symbol": "^CNXIT",
        "tickers": [
            "TCS",
            "INFY",
            "WIPRO",
            "HCLTECH",
            "TECHM",
            "LTIM",
            "MPHASIS",
            "PERSISTENT",
            "COFORGE",
            "OFSS",
        ],
    },
    {
        "name": "Nifty Auto",
        "index_symbol": "^CNXAUTO",
        "tickers": [
            "MARUTI",
            "M&M",
            "TATAMOTORS",
            "BAJAJ-AUTO",
            "EICHERMOT",
            "HEROMOTOCO",
            "TVSMOTOR",
            "MOTHERSON",
        ],
    },
    {
        "name": "Nifty Pharma",
        "index_symbol": "^CNXPHARMA",
        "tickers": [
            "SUNPHARMA",
            "DRREDDY",
            "CIPLA",
            "DIVISLAB",
            "LUPIN",
            "AUROPHARMA",
            "TORNTPHARM",
            "ALKEM",
        ],
    },
    {
        "name": "Nifty FMCG",
        "index_symbol": "^CNXFMCG",
        "tickers": [
            "HINDUNILVR",
            "ITC",
            "NESTLEIND",
            "BRITANNIA",
            "DABUR",
            "MARICO",
            "COLPAL",
            "GODREJCP",
        ],
    },
    {
        "name": "Nifty Energy",
        "index_symbol": "^CNXENERGY",
        "tickers": [
            "RELIANCE",
            "ONGC",
            "BPCL",
            "HPCL",
            "IOC",
            "GAIL",
            "PETRONET",
            "OIL",
        ],
    },
    {
        "name": "Nifty Metal",
        "index_symbol": "^CNXMETAL",
        "tickers": [
            "TATASTEEL",
            "HINDALCO",
            "JSWSTEEL",
            "VEDL",
            "SAIL",
            "NMDC",
            "NATIONALUM",
        ],
    },
    {
        "name": "Nifty Financial Services",
        "index_symbol": "^CNXFINANCE",
        "tickers": [
            "BAJFINANCE",
            "BAJAJFINSV",
            "HDFCLIFE",
            "SBILIFE",
            "ICICIPRULI",
            "CHOLAFIN",
            "MUTHOOTFIN",
            "LICHSGFIN",
        ],
    },
    {
        "name": "Nifty Infrastructure",
        "index_symbol": "^CNXINFRA",
        "tickers": [
            "LARSEN",
            "ULTRACEMCO",
            "ADANIPORTS",
            "SHREECEM",
            "AMBUJACEMENT",
            "ACC",
            "GMRINFRA",
        ],
    },
]

# NSE ticker -> sector index symbol, e.g. "RELIANCE" -> "^CNXENERGY"
SECTOR_INDEX: dict[str, str] = {
    ticker: sector["index_symbol"] for sector in SECTORS for ticker in sector["tickers"]
}

# sector index symbol -> display name, e.g. "^CNXENERGY" -> "Nifty Energy"
SECTOR_NAMES: dict[str, str] = {sector["index_symbol"]: sector["name"] for sector in SECTORS}
SECTOR_NAMES["^NSEI"] = "Nifty 50"  # fallback index for tickers with no sector mapping
