from __future__ import annotations

import json
import re

from dalal_agents.models import DebateStance, DebateTranscript, DebateTurn, Signal

_LLM = object


def _extract_json(text: str) -> dict:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in LLM response. First 300 chars:\n{text[:300]}")


def _format_transcript(transcript: DebateTranscript) -> str:
    if not transcript.turns:
        return "(No previous turns — you are opening the debate.)"
    lines: list[str] = []
    for turn in transcript.turns:
        lines.append(
            f"\n--- Turn {turn.turn_number}: {turn.speaker} [{turn.stance}] ---"
        )
        lines.append(turn.argument)
        lines.append("Key Points:")
        for pt in turn.key_points:
            lines.append(f"  • {pt}")
    return "\n".join(lines)


async def generate_debate_turn(
    llm: _LLM,
    agent_name: str,
    stance: DebateStance,
    system_prompt: str,
    context: str,
    transcript: DebateTranscript,
    turn_number: int,
) -> DebateTurn:
    transcript_text = _format_transcript(transcript)

    user_message = (
        f"=== DEBATE CONTEXT ===\n{context}\n\n"
        f"=== DEBATE TRANSCRIPT SO FAR ===\n{transcript_text}\n\n"
        f"=== YOUR TASK — TURN {turn_number} ===\n"
        f"You are **{agent_name}** (stance: {stance}).\n"
        "Produce a rigorous argument for your stance:\n"
        "  • Use SPECIFIC numbers from the context (cite RSI, ROCE, pledge %, etc.).\n"
        "  • Directly rebut any opposing arguments already in the transcript above.\n"
        "  • Do not be vague — general claims without data do not win debates.\n\n"
        "Respond ONLY with a JSON object (no surrounding text):\n"
        '{"argument": "<your full argument>", '
        '"key_points": ["<point 1>", "<point 2>", "<point 3>"]}'
    )

    response = await llm.call(
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        max_tokens=2048,
    )

    text = response.text or ""
    data = _extract_json(text)

    return DebateTurn(
        speaker=agent_name,
        stance=stance,
        argument=data["argument"],
        key_points=data.get("key_points", []),
        turn_number=turn_number,
    )


_FACILITATOR_SYSTEM = """\
You are an impartial Investment Committee Chairman reviewing a structured debate
about an Indian equity.

Your job: determine which side presented the BETTER-EVIDENCED argument.
Evidence and specific data beats rhetoric. A claim backed by a number from the
analyst report beats an unsupported assertion.

You must output a JSON object and NOTHING else:
{
  "facilitator_verdict": "<2-3 sentences: which side won and WHY, citing specific evidence>",
  "winning_stance": "<BULLISH | BEARISH | NEUTRAL | RISKY | SAFE>",
  "consensus_signal": "<STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "key_opportunities": ["<opportunity 1>", "<opportunity 2>", "<opportunity 3>"]
}
"""


async def run_facilitator(
    llm: _LLM,
    transcript: DebateTranscript,
    context: str,
) -> DebateTranscript:
    transcript_text = _format_transcript(transcript)

    user_message = (
        f"=== DEBATE TOPIC ===\n{transcript.topic}\n\n"
        f"=== DEBATE CONTEXT ===\n{context}\n\n"
        f"=== COMPLETE TRANSCRIPT ===\n{transcript_text}\n\n"
        "=== YOUR VERDICT ===\n"
        "Deliver your impartial ruling. Which side had the stronger evidence?\n"
        "Output ONLY the JSON object described in your instructions."
    )

    response = await llm.call(
        system=_FACILITATOR_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
        tools=None,
        max_tokens=1024,
    )

    data = _extract_json(response.text or "")

    transcript.facilitator_verdict = data.get("facilitator_verdict")
    transcript.winning_stance       = DebateStance(data["winning_stance"])
    transcript.consensus_signal     = Signal(data["consensus_signal"])
    transcript.key_risks            = data.get("key_risks", [])
    transcript.key_opportunities    = data.get("key_opportunities", [])

    return transcript
