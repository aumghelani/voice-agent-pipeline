"""Runnable example: a voice agent that can call tools.

Fully mocked (no mic/speakers/model downloads) so it demonstrates the tool
loop shape end-to-end. Run with:

    python -m examples.assistant_with_tools
"""

from __future__ import annotations

import json

from voiceagents.agent.memory import ConversationMemory
from voiceagents.llm import Message, ScriptedLLM
from voiceagents.tools import ToolRegistry, run_tool_loop
from voiceagents.tools.builtin import register_builtin_tools


def main() -> None:
    registry = ToolRegistry()
    register_builtin_tools(registry)

    # ScriptedLLM stands in for a real model: first call plays "router",
    # picking the calculator tool; second call plays "summarizer", turning
    # the raw result into a natural-language reply. A real LLM would do both
    # roles itself, driven by the prompts run_tool_loop sends it.
    llm = ScriptedLLM(
        replies=[
            json.dumps({"tool": "calculator", "arguments": {"expression": "12 * 7"}}),
            "Twelve times seven is 84.",
        ]
    )

    memory = ConversationMemory(system_prompt="You are a helpful voice assistant with tools.")
    memory.add_user_turn("What's 12 times 7?")

    result = run_tool_loop(llm, registry, memory.to_messages())
    memory.add_assistant_turn(result.reply)

    print(f"User: What's 12 times 7?")
    print(f"Tool used: {result.tool_used} -> {result.tool_result}")
    print(f"Assistant: {result.reply}")


if __name__ == "__main__":
    main()
