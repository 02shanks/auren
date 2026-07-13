"""Router

Intent categories are a *safety net*, not response templates. We always compute a
deterministic intent classification independent of the backend and use it to:
  1. enforce guardrail intents (defense in depth — a jailbreak is refused even if a
     future model would comply), and
  2. force-include the base tool set for compound queries, so an under-tooling model
     still gets the tools a request like "what should I study this week?" needs.

The configured client's own tool choices are unioned on top; nothing here writes the
final answer.
"""

from typing import Any

from src.llm.base import LLMClient, Selection
from src.llm.deterministic import DeterministicClient


def route(
    query: str,
    student_id: str,
    client: LLMClient,
    tools: list[dict],
    repo: Any,
    config: dict,
) -> tuple[Selection, list[str]]:
    safety = DeterministicClient(config, repo).select_tools(query, student_id, tools)
    if safety.direct_answer is not None:
        return safety, [f"safety-net intent: {safety.intent}"]

    primary = client.select_tools(query, student_id, tools)
    if primary.direct_answer is not None:
        return primary, []

    names = {c.name for c in primary.tool_calls}
    merged = list(primary.tool_calls)
    forced: list[str] = []
    for c in safety.tool_calls:
        if c.name not in names:
            merged.append(c)
            forced.append(c.name)

    intent = safety.intent if safety.intent != "unknown" else primary.intent
    selection = Selection(
        tool_calls=merged,
        intent=intent,
        focus_topic=safety.focus_topic or primary.focus_topic,
        focus_subject=safety.focus_subject or primary.focus_subject,
    )
    notes = [f"force-included {forced}"] if forced else []
    return selection, notes
