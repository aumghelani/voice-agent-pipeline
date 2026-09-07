"""Tool registry: name -> (argument schema, callable).

Tools are plain Python callables validated against a Pydantic model. The
registry exports JSON Schema for each tool so it can be described to an LLM
(either via native provider tool-calling, or embedded in a prompt for
providers without native support).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T", bound=BaseModel)


class ToolError(Exception):
    """Raised when a tool is unknown or its arguments fail validation."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]

    def json_schema(self) -> dict:
        return TypeAdapter(self.args_model).json_schema()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        args_model: type[BaseModel],
        fn: Callable[[BaseModel], Any],
        *,
        description: str = "",
    ) -> None:
        if name in self._tools:
            raise ToolError(f"Tool {name!r} is already registered")
        self._tools[name] = ToolSpec(
            name=name, description=description, args_model=args_model, fn=fn
        )

    def schema_list(self) -> list[dict]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.json_schema(),
            }
            for spec in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def call(self, name: str, raw_arguments: dict) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"Unknown tool: {name!r}. Known tools: {self.names()}")
        try:
            args = spec.args_model.model_validate(raw_arguments)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for tool {name!r}: {exc}") from exc
        return spec.fn(args)
