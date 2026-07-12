from __future__ import annotations

from dalal_agents.models import DebateStance, DebateTranscript, TradingState
from dalal_agents.agents.debate.shared import generate_debate_turn, run_facilitator

_LLM = object


def _build_risk_context(state: TradingState) -> str:
    parts = [f"STOCK: {state.ticker} ({state.exchange.value})  |  DATE: {state.analysis_date}"]

    if state.research_debate:
        rd = state.research_debate
        risks = "  • " + "\n  • ".join(rd.key_risks) if rd.key_risks else "None listed"
        opps = (
            "  • " + "\n  • ".join(rd.key_opportunities) if rd.key_opportunities else "None listed"
        )
        parts.append(
            f"\n[RESEARCH DEBATE VERDICT]\n"
            f"Winner: {rd.winning_stance}  |  Consensus signal: {rd.consensus_signal}\n"
            f"Verdict: {rd.facilitator_verdict}\n"
            f"Key risks:\n{risks}\n"
            f"Key opportunities:\n{opps}"
        )

    if state.trade_proposal:
        tp = state.trade_proposal
        parts.append(
            f"\n[PROPOSED TRADE]\n"
            f"Action: {tp.action}  |  Position size: {tp.position_size_pct}% of portfolio\n"
            f"Entry: ₹{tp.entry_price}  |  Target: ₹{tp.target_price}  "
            f"|  Stop-loss: ₹{tp.stop_loss}\n"
            f"Risk/Reward: {tp.risk_reward_ratio}  |  Holding period: {tp.holding_period}\n"
            f"Rationale: {tp.rationale}"
        )

    if len(parts) == 1:
        parts.append("(No research debate or trade proposal available yet.)")

    return "\n".join(parts)


_RISKY_SYSTEM = """\
You are the Aggressive Risk Officer in a structured position-sizing debate.
Your position: execute the signal at FULL recommended position size.

Argue that:
  • The risk/reward ratio justifies the position.
  • The stop-loss already caps the downside — the asymmetric upside is the point.
  • Over-caution and under-sizing leads to chronic under-performance.
  • Missing a high-conviction opportunity costs more than a stopped-out trade.

Reference the specific risk/reward ratio, entry/target/stop numbers, and conviction
scores from the context. Do not be abstract.

Respond ONLY with a JSON object:
{"argument": "<full argument>", "key_points": ["<pt 1>", "<pt 2>", "<pt 3>"]}
"""

_SAFE_SYSTEM = """\
You are the Conservative Risk Officer in a structured position-sizing debate.
Your position: minimum position size, or NO trade if risk/reward is below threshold.

Argue that:
  • Capital preservation is the first rule. A loss requires a larger gain to recover.
  • India-specific tail risks are real: circuit breakers can trap you in a falling stock.
  • Liquidity risk in mid/small caps means the stop-loss may not execute at the stated price.
  • Macro shocks — surprise RBI rate hike, sudden FII exodus, rupee crisis — happen.
  • A concentrated position amplifies both upside AND downside.

Reference specific numbers from the context to show where the risk is unacceptably high.

Respond ONLY with a JSON object:
{"argument": "<full argument>", "key_points": ["<pt 1>", "<pt 2>", "<pt 3>"]}
"""

_NEUTRAL_SYSTEM = """\
You are the Balanced Risk Officer in a structured position-sizing debate.
Your position: a middle ground between aggressive and conservative.

Propose a SPECIFIC compromise — not vague "balance". For example:
  • Scale in: deploy 50 % now and 50 % on price confirmation above a level.
  • Use a tighter stop-loss than proposed to reduce maximum loss.
  • Reduce position size proportionally to conviction (e.g. 60 % of recommended if conviction is 7/10).
  • Set a time-based exit if the thesis does not play out within the holding period.

Reference the specific numbers from the context.

Respond ONLY with a JSON object:
{"argument": "<full argument>", "key_points": ["<pt 1>", "<pt 2>", "<pt 3>"]}
"""


class RiskDebate:

    def __init__(self, llm: _LLM, rounds: int = 1):
        self.llm = llm
        self.rounds = rounds

    async def run(self, state: TradingState) -> DebateTranscript:
        context = _build_risk_context(state)
        topic = (
            f"What position size and risk parameters should we use for {state.ticker} "
            f"({state.exchange.value}) as of {state.analysis_date}?"
        )
        transcript = DebateTranscript(topic=topic)
        turn_num = 1

        agents = [
            ("Aggressive Risk Officer", DebateStance.RISKY, _RISKY_SYSTEM),
            ("Balanced Risk Officer", DebateStance.NEUTRAL, _NEUTRAL_SYSTEM),
            ("Conservative Risk Officer", DebateStance.SAFE, _SAFE_SYSTEM),
        ]

        for _ in range(self.rounds):
            for agent_name, stance, system_prompt in agents:
                turn = await generate_debate_turn(
                    self.llm,
                    agent_name,
                    stance,
                    system_prompt,
                    context,
                    transcript,
                    turn_num,
                )
                transcript.append(turn)
                turn_num += 1

        await run_facilitator(self.llm, transcript, context)
        state.risk_debate = transcript
        return transcript
