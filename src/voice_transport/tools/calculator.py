"""Safe session-local arithmetic tool."""

from __future__ import annotations

import ast
import operator
from typing import Any

from .base import ToolDefinition, ToolResult

_CALCULATE = "transport__Calculate"
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> int | float:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    raise ValueError("expression contains an unsupported operation")


class CalculatorToolProvider:
    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                _CALCULATE,
                "Calculate an arithmetic expression locally. Always use this tool "
                "for mathematical operations; never do mental math.",
                {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != _CALCULATE:
            return ToolResult([{"type": "text", "text": f"Unknown tool: {name}"}], True)
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            return ToolResult(
                [{"type": "text", "text": "expression must be a string"}], True
            )
        try:
            result = _evaluate(ast.parse(expression, mode="eval").body)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as err:
            return ToolResult(
                [{"type": "text", "text": f"Calculation failed: {err}"}], True
            )
        return ToolResult([{"type": "text", "text": str(result)}])

    async def close(self) -> None:
        return None
