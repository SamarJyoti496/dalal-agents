from __future__ import annotations

from datetime import date

from dalal_agents.models import TradingState
from dalal_agents.tools import get_ohlcv


def _last_close(ticker_symbol: str, as_of_date: str) -> dict:
    df = get_ohlcv(ticker_symbol, date.fromisoformat(as_of_date))
    return {
        "ticker_symbol": ticker_symbol,
        "last_close_inr": round(float(df["Close"].iloc[-1]), 2),
        "date": str(df.index[-1].date()),
    }


def _analyst_signal_table(state: TradingState) -> str:
    rows: list[str] = []
    if state.technical_report:
        r = state.technical_report
        rows.append(
            f"  Technical:    {r.signal:<12} conviction {r.conviction}/10 | "
            f"trend={r.trend} | support=₹{r.support_level} | resistance=₹{r.resistance_level}"
        )
    else:
        rows.append("  Technical:    NOT AVAILABLE")

    if state.sentiment_report:
        r = state.sentiment_report
        rows.append(
            f"  Sentiment:    {r.signal:<12} conviction {r.conviction}/10 | "
            f"overall={r.overall_sentiment} | fii_dii={r.fii_dii_flow}"
        )
    else:
        rows.append("  Sentiment:    NOT AVAILABLE")

    if state.news_report:
        r = state.news_report
        rows.append(
            f"  News:         {r.signal:<12} conviction {r.conviction}/10 | "
            f"rbi={r.rbi_stance} | earnings_soon={r.earnings_upcoming}"
        )
    else:
        rows.append("  News:         NOT AVAILABLE")

    if state.fundamentals_report:
        r = state.fundamentals_report
        rows.append(
            f"  Fundamentals: {r.signal:<12} conviction {r.conviction}/10 | "
            f"roce={r.roce}% | promoter_pledge={r.promoter_pledge_pct}% | "
            f"fii_holding={r.fii_holding_pct}%"
        )
    else:
        rows.append("  Fundamentals: NOT AVAILABLE")

    return "\n".join(rows)


def _research_debate_summary(state: TradingState) -> str:
    if not state.research_debate:
        return "  (Research debate not yet run)"
    rd = state.research_debate
    losing_arg = ""
    if rd.turns and rd.winning_stance:
        for turn in reversed(rd.turns):
            if turn.stance != rd.winning_stance:
                losing_arg = f'\n  Strongest dissenting argument:\n  "{turn.argument[:300]}..."'
                break
    return (
        f"  Winner:    {rd.winning_stance}  |  Consensus signal: {rd.consensus_signal}\n"
        f"  Verdict:   {rd.facilitator_verdict}\n"
        f"  Key risks: {'; '.join(rd.key_risks[:3])}" + losing_arg
    )


def _trade_proposal_summary(state: TradingState) -> str:
    if not state.trade_proposal:
        return "  (Trade proposal not yet generated)"
    tp = state.trade_proposal
    return (
        f"  Action: {tp.action}  |  Size: {tp.position_size_pct}% of portfolio\n"
        f"  Entry:  ₹{tp.entry_price}  |  Target: ₹{tp.target_price}  "
        f"|  Stop-loss: ₹{tp.stop_loss}\n"
        f"  R/R:    {tp.risk_reward_ratio}  |  Holding period: {tp.holding_period}\n"
        f"  Rationale: {tp.rationale}"
    )


def _risk_debate_summary(state: TradingState) -> str:
    if not state.risk_debate:
        return "  (Risk debate not yet run)"
    rd = state.risk_debate
    return (
        f"  Winner:  {rd.winning_stance}\n"
        f"  Verdict: {rd.facilitator_verdict}\n"
        f"  Key risks: {'; '.join(rd.key_risks[:3])}"
    )


def _risk_assessment_summary(state: TradingState) -> str:
    if not state.risk_assessment:
        return "  (Risk assessment not yet run)"
    ra = state.risk_assessment
    return (
        f"  Approved action:    {ra.approved_action}\n"
        f"  Adjusted size:      {ra.adjusted_position_size_pct}%\n"
        f"  Adjusted stop-loss: ₹{ra.adjusted_stop_loss}\n"
        f"  Risk level:         {ra.risk_level}\n"
        f"  India VIX:          {ra.vix_india}\n"
        f"  Rationale: {ra.rationale}"
    )
