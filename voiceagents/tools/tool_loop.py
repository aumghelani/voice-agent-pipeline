"""Prompt-based tool-calling loop for LLM providers without native
function-calling.

Two-pass design:
1. Router pass: ask the LLM to emit exactly one JSON object
   `{"tool": <name>, "arguments": {...}}` (or `{"tool": null}` if no tool
   applies), given the registry's schema list in the system prompt.
2. Summarizer pass: feed the tool's raw output back to the LLM as a fresh
   turn and ask for a natural-language reply.

If the model's output isn't valid JSON, or names an unknown tool, or the
arguments fail validation, the loop degrades to treating the router pass
output as a direct reply rather than raising - a router misfire shouldn't
break the conversation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from voiceagents.llm.base import LLMProvider, Message
from voiceagents.tools.registry import ToolError, ToolRegistry

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class ToolLoopResult:
    reply: str
    tool_used: str | None
    tool_result: object | None


def _extract_json_object(text: str) -> dict | None:
    fenced = _JSON_FENCE.search(text)
    candidate = fenced.group(1) if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to slicing the first {...} span, in case the model added
    # surrounding prose despite instructions not to.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


def _router_system_prompt(registry: ToolRegistry) -> str:
    schemas = json.dumps(registry.schema_list(), indent=2)
    return (
        "You can call one tool to help answer the user. Available tools:\n"
        f"{schemas}\n\n"
        'Respond with exactly one JSON object: {"tool": <tool name or null>, '
        '"arguments": {...}}. Use null when no tool is needed. '
        "Do not include any other text."
    )


def run_tool_loop(
    llm: LLMProvider,
    registry: ToolRegistry,
    conversation: list[Message],
    *,
    summarizer_system_prompt: str = (
        "Turn the following tool result into a short, natural-language reply "
        "to the user's question. Do not mention that you used a tool."
    ),
) -> ToolLoopResult:
    router_messages = [Message("system", _router_system_prompt(registry)), *conversation]
    router_output = llm.complete(router_messages)

    decision = _extract_json_object(router_output)
    if decision is None:
        # Model didn't follow the JSON protocol; treat its raw output as the reply.
        return ToolLoopResult(reply=router_output, tool_used=None, tool_result=None)

    tool_name = decision.get("tool")
    if not tool_name:
        return ToolLoopResult(reply=router_output, tool_used=None, tool_result=None)

    arguments = decision.get("arguments") or {}
    try:
        tool_result = registry.call(tool_name, arguments)
    except ToolError as exc:
        return ToolLoopResult(reply=f"I couldn't complete that: {exc}", tool_used=tool_name, tool_result=None)

    summarizer_messages = [
        Message("system", summarizer_system_prompt),
        Message("user", f"Tool `{tool_name}` returned: {tool_result!r}"),
    ]
    reply = llm.complete(summarizer_messages)
    return ToolLoopResult(reply=reply, tool_used=tool_name, tool_result=tool_result)
