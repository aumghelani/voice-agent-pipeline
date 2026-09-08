"""A few small example tools to exercise the registry and tool loop."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from voiceagents.tools.registry import ToolRegistry


class CalculatorArgs(BaseModel):
    expression: str


_ALLOWED_CHARS = set("0123456789.+-*/() ")


def _calculate(args: CalculatorArgs) -> float:
    expr = args.expression
    if not expr or not set(expr) <= _ALLOWED_CHARS:
        raise ValueError(f"Unsupported characters in expression: {expr!r}")
    # A restricted eval: only digits/operators/parens are permitted above,
    # and no names/builtins are exposed, so this can't execute arbitrary code.
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307


class CurrentTimeArgs(BaseModel):
    pass


def _current_time(_args: CurrentTimeArgs) -> str:
    return datetime.now(timezone.utc).isoformat()


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        "calculator",
        CalculatorArgs,
        _calculate,
        description="Evaluate a basic arithmetic expression, e.g. '2 + 3 * 4'.",
    )
    registry.register(
        "current_time",
        CurrentTimeArgs,
        _current_time,
        description="Get the current UTC time in ISO 8601 format.",
    )
