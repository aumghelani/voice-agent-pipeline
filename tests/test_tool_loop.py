import json

from voiceagents.llm import Message, ScriptedLLM
from voiceagents.tools import ToolRegistry, run_tool_loop
from voiceagents.tools.builtin import register_builtin_tools


def test_tool_loop_routes_to_tool_and_summarizes():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = ScriptedLLM(
        replies=[
            json.dumps({"tool": "calculator", "arguments": {"expression": "2 + 2"}}),
            "That's 4.",
        ]
    )

    result = run_tool_loop(llm, registry, [Message("user", "what's 2 + 2?")])

    assert result.tool_used == "calculator"
    assert result.tool_result == 4
    assert result.reply == "That's 4."


def test_tool_loop_handles_no_tool_needed():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = ScriptedLLM(replies=[json.dumps({"tool": None})])

    result = run_tool_loop(llm, registry, [Message("user", "hi there")])

    assert result.tool_used is None
    assert result.reply == json.dumps({"tool": None})


def test_tool_loop_extracts_json_from_markdown_fence():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    fenced = "```json\n" + json.dumps({"tool": "current_time", "arguments": {}}) + "\n```"
    llm = ScriptedLLM(replies=[fenced, "It's noon UTC."])

    result = run_tool_loop(llm, registry, [Message("user", "what time is it?")])

    assert result.tool_used == "current_time"
    assert result.reply == "It's noon UTC."


def test_tool_loop_falls_back_to_raw_output_on_invalid_json():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = ScriptedLLM(replies=["Sure, I can help with that!"])

    result = run_tool_loop(llm, registry, [Message("user", "hi")])

    assert result.tool_used is None
    assert result.reply == "Sure, I can help with that!"


def test_tool_loop_reports_error_for_unknown_tool_without_raising():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = ScriptedLLM(replies=[json.dumps({"tool": "nonexistent", "arguments": {}})])

    result = run_tool_loop(llm, registry, [Message("user", "hi")])

    assert result.tool_used == "nonexistent"
    assert result.tool_result is None
    assert "couldn't complete" in result.reply


def test_tool_loop_reports_error_for_invalid_arguments_without_raising():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = ScriptedLLM(
        replies=[json.dumps({"tool": "calculator", "arguments": {"expression": "bad$expr"}})]
    )

    result = run_tool_loop(llm, registry, [Message("user", "compute bad$expr")])

    assert result.tool_used == "calculator"
    assert "couldn't complete" in result.reply


def test_tool_loop_sends_router_system_prompt_with_tool_schemas():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = ScriptedLLM(replies=[json.dumps({"tool": None})])

    run_tool_loop(llm, registry, [Message("user", "hi")])

    router_call = llm.received_messages[0]
    assert router_call[0].role == "system"
    assert "calculator" in router_call[0].content
    assert "current_time" in router_call[0].content
