import pytest

from voiceagents.tools import ToolError, ToolRegistry
from voiceagents.tools.builtin import register_builtin_tools


def test_calculator_evaluates_expression():
    registry = ToolRegistry()
    register_builtin_tools(registry)

    result = registry.call("calculator", {"expression": "2 + 3 * 4"})

    assert result == 14


def test_calculator_rejects_disallowed_characters():
    registry = ToolRegistry()
    register_builtin_tools(registry)

    with pytest.raises(ToolError):
        registry.call("calculator", {"expression": "__import__('os').system('echo hi')"})


def test_calculator_rejects_empty_expression():
    registry = ToolRegistry()
    register_builtin_tools(registry)

    with pytest.raises(ToolError):
        registry.call("calculator", {"expression": ""})


def test_current_time_returns_iso_string():
    registry = ToolRegistry()
    register_builtin_tools(registry)

    result = registry.call("current_time", {})

    assert "T" in result
    assert result.endswith("+00:00")


def test_register_builtin_tools_registers_expected_names():
    registry = ToolRegistry()

    register_builtin_tools(registry)

    assert set(registry.names()) == {"calculator", "current_time"}
