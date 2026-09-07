import pytest
from pydantic import BaseModel

from voiceagents.tools import ToolError, ToolRegistry


class AddArgs(BaseModel):
    a: float
    b: float


def add(args: AddArgs) -> float:
    return args.a + args.b


def test_register_and_call_tool():
    registry = ToolRegistry()
    registry.register("add", AddArgs, add, description="Add two numbers")

    result = registry.call("add", {"a": 2, "b": 3})

    assert result == 5


def test_call_unknown_tool_raises_tool_error():
    registry = ToolRegistry()

    with pytest.raises(ToolError, match="Unknown tool"):
        registry.call("missing", {})


def test_call_with_invalid_arguments_raises_tool_error():
    registry = ToolRegistry()
    registry.register("add", AddArgs, add)

    with pytest.raises(ToolError, match="Invalid arguments"):
        registry.call("add", {"a": "not a number", "b": 1})


def test_register_duplicate_name_raises():
    registry = ToolRegistry()
    registry.register("add", AddArgs, add)

    with pytest.raises(ToolError, match="already registered"):
        registry.register("add", AddArgs, add)


def test_schema_list_includes_name_description_and_json_schema():
    registry = ToolRegistry()
    registry.register("add", AddArgs, add, description="Add two numbers")

    schemas = registry.schema_list()

    assert len(schemas) == 1
    assert schemas[0]["name"] == "add"
    assert schemas[0]["description"] == "Add two numbers"
    assert schemas[0]["parameters"]["properties"].keys() == {"a", "b"}


def test_names_lists_registered_tools():
    registry = ToolRegistry()
    registry.register("add", AddArgs, add)

    assert registry.names() == ["add"]


def test_call_coerces_compatible_types():
    registry = ToolRegistry()
    registry.register("add", AddArgs, add)

    result = registry.call("add", {"a": "2", "b": "3"})

    assert result == 5.0
