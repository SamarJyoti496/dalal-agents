from __future__ import annotations

import time

from dalal_agents.agents.debate import RiskDebate
from dalal_agents.agents.trading.fund_manager import FundManagerAgent
from dalal_agents.agents.trading.risk_assessment import RiskAssessmentAgent
from dalal_agents.agents.trading.trader import TraderAgent
from dalal_agents.models import TradingState


async def run_trading_stage(
    state: TradingState,
    llm,
    quiet: bool = False,
    memory_context: str = "",
    progress_callback=None,
) -> TradingState:
    """
    Stage II of the DalalAgents pipeline.

    Sequence:
      1. TraderAgent         → state.trade_proposal
      2. RiskDebate          → state.risk_debate
      3. RiskAssessmentAgent → state.risk_assessment
      4. FundManagerAgent    → state.final_decision

    Each sub-step fires progress_callback("stage_start"/"stage_done", stage=<name>)
    so the TUI can display them as separate progress rows.
    """
    _p = (lambda *a, **k: None) if quiet else print
    _cb = progress_callback or (lambda *a, **k: None)
    t_start = time.perf_counter()

    _cb("stage_start", stage="Trader")
    _p(f"[Stage II / Step 1] TraderAgent running for {state.ticker}...")
    trader = TraderAgent(llm=llm)
    proposal = await trader.run(state)
    state.trade_proposal = proposal
    _p(
        f"[Stage II / Step 1] TraderAgent done — action={proposal.action}, "
        f"size={proposal.position_size_pct}%, R/R={proposal.risk_reward_ratio}"
    )
    _cb(
        "stage_done",
        stage="Trader",
        action=str(proposal.action),
        size=proposal.position_size_pct,
        summary=proposal.rationale,
    )

    _cb("stage_start", stage="Risk Debate")
    _p(f"[Stage II / Step 2] RiskDebate running...")
    risk_debate = RiskDebate(llm=llm, rounds=1)
    await risk_debate.run(state)
    _p(
        f"[Stage II / Step 2] RiskDebate done — "
        f"winner={state.risk_debate.winning_stance if state.risk_debate else 'N/A'}"
    )
    _cb(
        "stage_done",
        stage="Risk Debate",
        winner=str(state.risk_debate.winning_stance) if state.risk_debate else None,
        summary=state.risk_debate.facilitator_verdict if state.risk_debate else None,
    )

    _cb("stage_start", stage="Risk Assessment")
    _p(f"[Stage II / Step 3] RiskAssessmentAgent running...")
    risk_agent = RiskAssessmentAgent(llm=llm)
    risk_assessment = await risk_agent.run(state)
    state.risk_assessment = risk_assessment
    _p(
        f"[Stage II / Step 3] RiskAssessmentAgent done — "
        f"approved={risk_assessment.approved_action}, "
        f"adjusted_size={risk_assessment.adjusted_position_size_pct}%, "
        f"risk={risk_assessment.risk_level}"
    )
    _cb(
        "stage_done",
        stage="Risk Assessment",
        action=str(risk_assessment.approved_action),
        size=risk_assessment.adjusted_position_size_pct,
        risk_level=str(risk_assessment.risk_level),
        summary=risk_assessment.rationale,
    )

    _cb("stage_start", stage="Fund Manager")
    _p(f"[Stage II / Step 4] FundManagerAgent running...")
    fund_manager = FundManagerAgent(llm=llm, memory_context=memory_context)
    final_decision = await fund_manager.run(state)
    state.final_decision = final_decision
    elapsed = round(time.perf_counter() - t_start, 1)
    _p(
        f"[Stage II / Step 4] FundManagerAgent done — "
        f"action={final_decision.action}, "
        f"size={final_decision.position_size_pct}% "
        f"(Stage II total: {elapsed}s)"
    )
    _cb(
        "stage_done",
        stage="Fund Manager",
        action=str(final_decision.action),
        size=final_decision.position_size_pct,
        summary=final_decision.rationale,
    )

    return state
