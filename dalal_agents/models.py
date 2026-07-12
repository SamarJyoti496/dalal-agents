from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


class Signal(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DebateStance(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    RISKY = "RISKY"
    SAFE = "SAFE"


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    result_summary: str
    called_at: datetime = Field(default_factory=datetime.now)


class TechnicalReport(BaseModel):
    ticker: str
    exchange: Exchange
    as_of_date: date
    rsi_14: Optional[float] = None
    macd_signal: Optional[str] = None  # e.g. "bullish_crossover", "bearish", "neutral"
    adx: Optional[float] = None
    bb_position: Optional[str] = None  # "above_upper" | "below_lower" | "inside"
    ema_20: Optional[float] = None  # INR
    ema_50: Optional[float] = None  # INR
    ema_20_vs_50: Optional[str] = None  # "golden_cross" | "death_cross" | "unknown"
    atr_pct: Optional[float] = None  # ATR as % of price
    vwap: Optional[float] = None  # INR
    vwap_position: Optional[str] = None  # "above" | "below"
    trend: str
    support_level: Optional[float] = None  # INR
    resistance_level: Optional[float] = None  # INR
    signal: Signal
    conviction: int = Field(ge=1, le=10)
    summary: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class SentimentReport(BaseModel):
    ticker: str
    as_of_date: date
    period_days: int = 7
    reddit_post_count: Optional[int] = None
    reddit_sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    twitter_mention_count: Optional[int] = None
    news_sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    fii_dii_flow: Optional[str] = None
    overall_sentiment: str
    signal: Signal
    conviction: int = Field(ge=1, le=10)
    summary: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class NewsReport(BaseModel):
    ticker: str
    as_of_date: date
    period_days: int = 7
    rbi_stance: Optional[str] = None  # "hawkish" | "dovish" | "neutral"
    budget_impact: Optional[str] = None
    sebi_news: Optional[str] = None
    sector_policy: Optional[str] = None
    top_headlines: list[str] = Field(default_factory=list)
    earnings_upcoming: bool = False
    promoter_news: Optional[str] = None
    signal: Signal
    conviction: int = Field(ge=1, le=10)
    summary: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class FundamentalsReport(BaseModel):
    ticker: str
    exchange: Exchange
    as_of_date: date
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    market_cap_cr: Optional[float] = None  # market cap in crores INR
    roe: Optional[float] = None
    roce: Optional[float] = None  # more important than ROE for Indian stocks
    net_profit_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    profit_growth_yoy: Optional[float] = None
    promoter_holding_pct: Optional[float] = None  # falling trend is a warning sign
    promoter_pledge_pct: Optional[float] = None  # above 30% is a red flag
    fii_holding_pct: Optional[float] = None
    dii_holding_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    signal: Signal
    conviction: int = Field(ge=1, le=10)
    summary: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class DebateTurn(BaseModel):
    speaker: str  # e.g. "Bull Researcher"
    stance: DebateStance
    argument: str
    key_points: list[str]  # 3–5 bullet points
    turn_number: int
    timestamp: datetime = Field(default_factory=datetime.now)


class DebateTranscript(BaseModel):
    topic: str
    turns: list[DebateTurn] = Field(default_factory=list)
    facilitator_verdict: Optional[str] = None
    winning_stance: Optional[DebateStance] = None
    consensus_signal: Optional[Signal] = None
    key_risks: list[str] = Field(default_factory=list)
    key_opportunities: list[str] = Field(default_factory=list)

    def append(self, turn: DebateTurn) -> None:
        self.turns.append(turn)


class TradeProposal(BaseModel):
    ticker: str
    exchange: Exchange
    as_of_date: date
    action: Signal
    position_size_pct: float = Field(ge=0.0, le=100.0)
    entry_price: Optional[float] = None  # INR
    target_price: Optional[float] = None  # INR
    stop_loss: Optional[float] = None  # INR
    risk_reward_ratio: Optional[float] = None
    holding_period: Literal["intraday", "swing_1w", "positional_1m", "investment_6m"]
    rationale: str
    key_assumptions: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    ticker: str
    as_of_date: date
    current_portfolio_exposure_pct: float = 0.0
    sector_exposure_pct: float = 0.0
    max_allowed_exposure_pct: float = 10.0
    nifty_trend: Optional[str] = None
    vix_india: Optional[float] = None  # above 20 = high fear; above 30 = extreme
    circuit_breaker_risk: bool = False  # True if stock is near its daily price limit
    approved_action: Signal
    adjusted_position_size_pct: float
    adjusted_stop_loss: Optional[float] = None  # INR
    risk_level: RiskLevel
    rationale: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class FinalDecision(BaseModel):
    ticker: str
    exchange: Exchange
    as_of_date: date
    action: Signal
    position_size_pct: float
    entry_price: Optional[float] = None  # INR
    target_price: Optional[float] = None  # INR
    stop_loss: Optional[float] = None  # INR
    rationale: str  # plain English, readable by a non-expert
    dissenting_view: Optional[str] = None  # strongest argument from the losing debate side
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    pipeline_duration_seconds: float = 0.0
    decided_at: datetime = Field(default_factory=datetime.now)


class TradingState(BaseModel):
    ticker: str
    exchange: Exchange = Exchange.NSE
    analysis_date: date
    ticker_symbol: str = ""

    # Stage outputs — populated as the pipeline progresses
    technical_report: Optional[TechnicalReport] = None
    sentiment_report: Optional[SentimentReport] = None
    news_report: Optional[NewsReport] = None
    fundamentals_report: Optional[FundamentalsReport] = None
    research_debate: Optional[DebateTranscript] = None
    trade_proposal: Optional[TradeProposal] = None
    risk_debate: Optional[DebateTranscript] = None
    risk_assessment: Optional[RiskAssessment] = None
    final_decision: Optional[FinalDecision] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.ticker_symbol:
            suffix = ".NS" if self.exchange == Exchange.NSE else ".BO"
            self.ticker_symbol = self.ticker + suffix
