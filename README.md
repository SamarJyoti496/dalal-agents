# DalalAgents

DalalAgents is a multi-agent LLM system that analyses Indian stock market tickers (NSE/BSE) and produces a reasoned trade decision for any given date. It runs a five-stage pipeline — analyst team, research debate, trader, risk debate, fund manager — each stage implemented as an autonomous Claude agent with tool access. All decisions are persisted in a local SQLite database, enabling historical replay and backtesting without repeating API calls.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage I — Analyst Team  (parallel, asyncio.gather)                 │
│                                                                     │
│  TechnicalAnalyst  SentimentAnalyst  NewsAnalyst  FundamentalsAnal. │
│       ↓                 ↓                 ↓              ↓          │
│             4 × AnalystReport  →  TradingState blackboard           │
└─────────────────────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage II — Research Debate                                         │
│                                                                     │
│  Bull Researcher  ↔  Bear Researcher  →  Facilitator               │
│                           ↓                                         │
│            DebateTranscript  (winning_stance, consensus_signal)     │
└─────────────────────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage III — Trading Stage                                          │
│                                                                     │
│  1. TraderAgent       →  TradeProposal  (entry, target, stop-loss)  │
│  2. RiskDebate        →  Risky / Neutral / Safe deliberation        │
│  3. RiskAssessmentAgt →  RiskAssessment  (India VIX, position cap)  │
│  4. FundManagerAgent  →  FinalDecision  (action, size, rationale)   │
└─────────────────────────────────────────────────────────────────────┘
             ↓
         SQLite  →  dalal_agents.db
```

Every agent runs a **ReAct loop**: it can call tools (yfinance, screener.in, NewsAPI, Reddit), inspect results, then produce a Pydantic-typed JSON output. Stages share a single `TradingState` blackboard object, so each stage sees everything the earlier stages produced.

---

## What makes it different from the TradingAgents paper repo

- **No LangGraph, no LangChain.** The pipeline and agent loop are implemented in plain `asyncio` with the Anthropic Messages API directly. Every tool call, every message, every state transition is visible Python code.
- **Indian markets first.** Tickers resolve to `RELIANCE.NS` (NSE) or `RELIANCE.BO` (BSE) automatically. Market context uses `^NSEI` (Nifty 50) and `^INDIAVIX`. FII/DII flow data is fetched from NSE's own API.
- **Hard look-ahead bias guard.** Every data-fetching function calls `_check_lookahead(as_of_date)` as its very first line. If the requested date is in the future, the function raises `ValueError` and refuses to proceed — you cannot accidentally train on tomorrow's prices.
- **SQLite persistence, no cloud required.** Every pipeline run is stored as a compressed JSON blob in `dalal_agents.db`. Re-running the same ticker + date is a sub-second DB read. Backtests reuse cached decisions and never call the LLM twice for the same day.

---

## Installation

```bash
git clone https://github.com/SamarJyoti496/dalal-agents.git
cd dalal-agents

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Verify the environment:

```bash
python verify_setup.py
```

---

## API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...          # required for --provider claude (default)
GEMINI_API_KEY=...                    # required for --provider gemini
OPENAI_API_KEY=sk-...                 # required for --provider openai
OPENROUTER_API_KEY=sk-or-...         # required for --provider openrouter
NEWSAPI_KEY=...                       # optional – richer news for SentimentAgent
REDDIT_CLIENT_ID=...                  # optional – Reddit sentiment
REDDIT_SECRET=...
REDDIT_USER_AGENT=DalalAgents/1.0 by u/yourhandle
```

You only need the key for the provider you actually use. OpenRouter gives access to Claude, Gemini, GPT-4o, Llama, and many others through a single key — get one at https://openrouter.ai/keys. Without the optional news/Reddit keys, those analyst agents return gracefully degraded reports.

---

## Usage

### Analyse a single stock

```bash
python dalal.py run RELIANCE --date 2024-01-15
```

```
──────────────────────────────────────────────────────────────
FINAL DECISION — RELIANCE (NSE) 2024-01-15
──────────────────────────────────────────────────────────────
  Action:        BUY
  Position size: 7.0%  of portfolio
  Entry price:   ₹2,847.50
  Target price:  ₹3,050.00
  Stop-loss:     ₹2,740.00
  Decided at:    2024-01-15T14:32:11
  Pipeline time: 87s

  Rationale:
    Reliance Industries shows a strong technical breakout above the 200-day EMA.
    Fundamentals are robust with ROCE of 12.8% and zero promoter pledge.
    The research debate reached a clear bullish consensus with moderate key risks.
    Risk assessment sets position cap at 7% given India VIX at 14.2.
──────────────────────────────────────────────────────────────
```

### Backtest over a quarter

```bash
python dalal.py backtest TCS --start 2024-01-01 --end 2024-03-31 --capital 1000000
```

```
Backtest: TCS (NSE)
  Period:       2024-01-01 → 2024-03-31  (61 trading days)
  Capital:      ₹10,00,000

BACKTEST RESULTS — TCS
──────────────────────────────────────────────────────────────
  Initial capital:     ₹  10,00,000
  Final value:         ₹  11,24,300
  Cumulative return:       +12.43%
  Sharpe ratio:              1.847
  Max drawdown:             -4.21%
  Trading days:                 61
──────────────────────────────────────────────────────────────
```

### Review a stored decision in detail

```bash
python dalal.py show RELIANCE --date 2024-01-15
```

Shows the full final decision, all research debate turns (Bull vs Bear arguments + key points), and summaries from all four analyst agents — directly from the local SQLite cache, no API calls.

### Browse your analysis history

```bash
python dalal.py history RELIANCE   # table of all RELIANCE decisions
python dalal.py list               # all tickers ever analysed
```

---

## Running the test suite

```bash
# Fast smoke tests — no LLM calls, just imports and model validation
pytest tests/ -v -s -k "not pipeline and not trading_stage and not debate"

# Full integration tests — use real LLM calls (~₹25–50 per suite run)
pytest tests/test_pipeline.py -v -s
pytest tests/test_debate.py -v -s
pytest tests/test_trading_stage.py -v -s
```

---

## Project structure

```
dalal-agents/
├── dalal.py                    ← CLI entry point
├── dalal_agents/
│   ├── config.py               ← environment / constants
│   ├── models.py               ← all Pydantic models (TradingState, reports, decisions)
│   ├── tools.py                ← 8 data-fetching functions with look-ahead guards
│   ├── agent.py                ← LLM clients, BaseAgent ReAct loop, TechnicalAnalystAgent
│   ├── analyst_team.py         ← Sentiment / News / Fundamentals agents + AnalystTeam runner
│   ├── debate.py               ← ResearchDebate, RiskDebate, generate_debate_turn
│   ├── trading_stage.py        ← TraderAgent, RiskAssessmentAgent, FundManagerAgent
│   └── pipeline.py             ← SQLite persistence, run_pipeline, run_backtest
├── tests/
│   ├── test_pipeline.py        ← end-to-end pipeline integration tests
│   ├── test_debate.py          ← debate module integration tests
│   └── test_trading_stage.py   ← trading stage integration tests
├── verify_setup.py             ← environment check (14 packages + API key)
├── requirements.txt
└── .env                        ← not committed
```

---

## Disclaimer

DalalAgents is a research and educational project. Nothing it produces constitutes financial advice, investment recommendations, or a solicitation to buy or sell any security. Past backtest performance does not guarantee future results. Use it to learn about multi-agent LLM systems — not to make real trading decisions.
