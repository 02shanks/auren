"""Tool-call validation

Sits between "what the model asked to call" and "what actually runs":

- drops hallucinated tool names,
- keeps only arguments the tool declares and drops calls still missing required
  arguments,
- forces ``student_id`` to the active student on every call that takes one — the
  structural half of cross-student isolation, so no tool can be driven with another
  student's id even if a model emits one.
"""

from src.llm.base import Selection, ToolCall
from src.tools.registry import allowed_params, required_params, tool_names


def validate(
    selection: Selection, active_student_id: str, config: dict | None = None
) -> tuple[list[ToolCall], list[str]]:
    notes: list[str] = []
    valid: list[ToolCall] = []
    known = tool_names()
    for call in selection.tool_calls:
        if call.name not in known:
            notes.append(f"dropped hallucinated tool '{call.name}'")
            continue
        allowed = allowed_params(call.name)
        args = {k: v for k, v in (call.arguments or {}).items() if k in allowed}
        if "student_id" in allowed:
            if args.get("student_id") not in (None, "", active_student_id):
                notes.append(f"rewrote student_id -> active on '{call.name}'")
            args["student_id"] = active_student_id
        missing = [r for r in required_params(call.name) if not args.get(r)]
        if missing:
            notes.append(f"dropped '{call.name}': missing required {missing}")
            continue
        valid.append(ToolCall(call.name, args))
    return valid, notes
