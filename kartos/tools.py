"""Tiny tool registry for function calling. Add your own with @registry.tool(...)."""
from __future__ import annotations

import ast
import operator as op
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[dict, Callable[..., Any]]] = {}

    def tool(self, name: str, description: str, properties: dict | None = None, required: list[str] | None = None):
        def deco(fn):
            schema = {
                "name": name,
                "description": description,
                "input_schema": {"type": "object", "properties": properties or {}, "required": required or []},
            }
            self._tools[name] = (schema, fn)
            return fn
        return deco

    def schemas(self) -> list[dict]:
        return [s for s, _ in self._tools.values()]

    def __bool__(self) -> bool:
        return bool(self._tools)

    async def call(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"error: unknown tool {name}"
        try:
            result = self._tools[name][1](**args)
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        except Exception as e:  # tool errors go back to the model, not the call
            return f"error: {e}"


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow}


def _eval(n: ast.AST) -> float:
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -_eval(n.operand)
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    raise ValueError("unsupported expression")


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("get_current_time", "Get the current date and time.",
              {"timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata. Default UTC."}})
    def get_current_time(timezone: str = "UTC") -> str:
        return datetime.now(ZoneInfo(timezone)).strftime("%A, %d %B %Y, %I:%M %p %Z")

    @reg.tool("calculate", "Evaluate an arithmetic expression like '(12.5 * 4) / 3'.",
              {"expression": {"type": "string"}}, ["expression"])
    def calculate(expression: str) -> str:
        return str(_eval(ast.parse(expression, mode="eval").body))

    return reg
