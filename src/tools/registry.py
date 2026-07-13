"""Single tool registry

One place defines the tool set; from it we derive both the LLM-facing function
schemas and the name->callable dispatch table, so the two can never drift apart.
Dispatch tolerates unknown tool names and bad arguments
by returning a structured error rather than raising.
"""

from collections.abc import Callable
from typing import Any

from src.tools import (
    get_performance_summary,
    get_upcoming_tests,
    get_weak_topics,
    log_feedback,
    recommend_study_material,
)
from src.tools.base import ToolContext

_MODULES = [
    get_weak_topics,
    get_upcoming_tests,
    get_performance_summary,
    recommend_study_material,
    log_feedback,
]

TOOLS: dict[str, tuple[Callable[..., dict], dict]] = {
    mod.SPEC["function"]["name"]: (mod.run, mod.SPEC) for mod in _MODULES
}


def tool_specs() -> list[dict]:
    """OpenAI/Ollama-style function schemas for the LLM."""
    return [spec for _run, spec in TOOLS.values()]


def tool_names() -> set[str]:
    return set(TOOLS)


def allowed_params(name: str) -> set[str]:
    spec = TOOLS[name][1]
    return set(spec["function"]["parameters"].get("properties", {}))


def required_params(name: str) -> list[str]:
    spec = TOOLS[name][1]
    return list(spec["function"]["parameters"].get("required", []))


def dispatch(name: str, ctx: ToolContext, arguments: dict[str, Any] | None) -> dict:
    if name not in TOOLS:
        return {"error": "unknown_tool", "tool": name, "known_tools": sorted(tool_names())}
    args = {k: v for k, v in (arguments or {}).items() if k in allowed_params(name)}
    fn = TOOLS[name][0]
    try:
        return fn(ctx, **args)
    except Exception as exc:  # a genuine bug in a tool -> structured error, never a crash
        return {"error": "tool_exception", "tool": name, "detail": str(exc)}
