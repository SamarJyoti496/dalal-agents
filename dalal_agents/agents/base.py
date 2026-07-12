from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, ValidationError as PydanticValidationError

from dalal_agents.config import MAX_TOOL_ITERATIONS
from dalal_agents.models import TradingState, ToolCall

logger = logging.getLogger("dalal_agents.agents")


def _is_rate_limit(exc: Exception) -> bool:
    if "RateLimit" in type(exc).__name__:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


def _rate_limit_wait(exc: Exception) -> float:
    m = re.search(r"try again in (\d+\.?\d*)\s*s", str(exc), re.IGNORECASE)
    return float(m.group(1)) + 1.0 if m else 15.0


class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict
    fn: Callable


class BaseAgent(ABC):
    """
    Abstract ReAct agent. Subclasses declare their system_prompt, tools,
    output_schema, and user-message builder. The run() method handles the
    full Observe → Think → Act loop automatically.
    """

    def __init__(self, llm):
        self.llm = llm

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def tools(self) -> list[ToolDefinition]: ...

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]: ...

    @abstractmethod
    def _build_user_message(self, state: TradingState) -> str: ...

    def _as_anthropic_tool_specs(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self.tools
        ]

    @staticmethod
    def _format_assistant_tool_use(text: Optional[str], tool_calls: list[dict]) -> list[dict]:
        blocks: list[dict] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in tool_calls:
            blocks.append(
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
            )
        return blocks

    @staticmethod
    def _format_tool_results(results: list[dict]) -> list[dict]:
        return [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_use_id"],
                "content": json.dumps(r["result"], default=str),
            }
            for r in results
        ]

    async def _run_one_tool(self, td: ToolDefinition, input_dict: dict) -> Any:
        if asyncio.iscoroutinefunction(td.fn):
            return await td.fn(**input_dict)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: td.fn(**input_dict))

    async def _execute_tools(self, tc_list: list[dict]) -> list[dict]:
        tool_map = {t.name: t for t in self.tools}

        async def run_one(tc: dict) -> dict:
            td = tool_map.get(tc["name"])
            if td is None:
                return {"tool_use_id": tc["id"], "result": {"error": f"Unknown tool: {tc['name']}"}}
            try:
                result = await self._run_one_tool(td, tc["input"])
                return {"tool_use_id": tc["id"], "result": result}
            except Exception as exc:
                return {"tool_use_id": tc["id"], "result": {"error": str(exc)}}

        return list(await asyncio.gather(*[run_one(tc) for tc in tc_list]))

    async def run(self, state: TradingState) -> BaseModel:
        name = self.__class__.__name__
        messages: list[dict] = [{"role": "user", "content": self._build_user_message(state)}]
        tool_specs = self._as_anthropic_tool_specs()
        accumulated_calls: list[ToolCall] = []
        last_text = ""

        for _iteration in range(MAX_TOOL_ITERATIONS):
            logger.debug(
                "%s iteration %d/%d: calling LLM", name, _iteration + 1, MAX_TOOL_ITERATIONS
            )

            try:
                response = await self.llm.call(
                    system=self.system_prompt,
                    messages=messages,
                    tools=tool_specs or None,
                )
            except Exception as exc:
                if _is_rate_limit(exc):
                    wait = _rate_limit_wait(exc)
                    logger.warning("%s hit rate limit, waiting %.0fs", name, wait)
                    print(f"  [rate limit] waiting {wait:.0f}s before retry…")
                    await asyncio.sleep(wait)
                    continue
                logger.exception("%s LLM call failed", name)
                raise

            if response.tool_calls:
                logger.debug(
                    "%s iteration %d: called tools %s",
                    name,
                    _iteration + 1,
                    [tc["name"] for tc in response.tool_calls],
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_assistant_tool_use(
                            response.text, response.tool_calls
                        ),
                    }
                )
                results = await self._execute_tools(response.tool_calls)

                result_by_id = {r["tool_use_id"]: r["result"] for r in results}
                for tc in response.tool_calls:
                    raw = result_by_id.get(tc["id"], {})
                    accumulated_calls.append(
                        ToolCall(
                            tool_name=tc["name"],
                            arguments=tc["input"],
                            result_summary=json.dumps(raw, default=str)[:500],
                        )
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": self._format_tool_results(results),
                    }
                )
                continue

            text = response.text or ""
            last_text = text
            json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

            if not json_match:
                logger.debug(
                    "%s iteration %d: no ```json fence in response, retrying. Raw text:\n%s",
                    name,
                    _iteration + 1,
                    text[:2000],
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Good analysis. Now output a complete {self.output_schema.__name__} "
                            "as valid JSON inside a ```json\\n...\\n``` fence. "
                            "No extra text outside the fence."
                        ),
                    }
                )
                continue

            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError as exc:
                logger.error(
                    "%s iteration %d: malformed JSON: %s\n%s",
                    name,
                    _iteration + 1,
                    exc,
                    json_match.group(1)[:2000],
                )
                raise RuntimeError(
                    f"LLM produced malformed JSON: {exc}\n\n{json_match.group(1)}"
                ) from exc

            if isinstance(data, dict) and len(data) == 1:
                only_key = next(iter(data))
                if only_key.lower() == self.output_schema.__name__.lower():
                    data = data[only_key]

            data.setdefault("ticker", state.ticker)
            data.setdefault("as_of_date", str(state.analysis_date))
            data.setdefault("exchange", state.exchange.value)
            data["tool_calls"] = [tc.model_dump() for tc in accumulated_calls]

            try:
                return self.output_schema.model_validate(data)
            except PydanticValidationError as exc:
                problems: list[str] = []
                for e in exc.errors():
                    field = str(e["loc"][0]) if e["loc"] else "?"
                    if e["type"] == "missing":
                        problems.append(f"  - {field}: MISSING (required)")
                    elif e["type"] == "enum":
                        problems.append(f"  - {field}: {e['input']!r} is not valid — {e['msg']}")
                    else:
                        problems.append(f"  - {field}: got {e['input']!r} — {e['msg']}")
                if not problems:
                    raise
                logger.debug(
                    "%s iteration %d: validation errors, retrying:\n%s",
                    name,
                    _iteration + 1,
                    "\n".join(problems),
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your JSON had validation errors:\n" + "\n".join(problems) + "\n\n"
                            "Rules:\n"
                            "  • signal must be exactly one of: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL\n"
                            "  • conviction must be an integer 1–10 (not a word like 'Low')\n"
                            f"Output the corrected {self.output_schema.__name__} inside a "
                            "```json\\n...\\n``` fence. No extra text outside the fence."
                        ),
                    }
                )
                continue

        logger.error(
            "%s exhausted %d iterations. Last raw response:\n%s",
            name,
            MAX_TOOL_ITERATIONS,
            last_text[:2000],
        )
        raise RuntimeError(
            f"{name} exhausted {MAX_TOOL_ITERATIONS} iterations "
            "without producing a valid JSON response. See the log file for the last raw "
            "LLM response."
        )
