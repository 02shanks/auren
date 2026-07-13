"""Context-window audit (blueprint sec 8.4 / sec 14.4).

Two checks:
  * ``audit_payload`` — one turn's assembled context: no other student's data, the payload
    is scoped to the active student, it stays within the char budget, and the mastery/session
    slices respect their caps.
  * ``audit_session_growth`` — across a multi-turn session the context must stay roughly flat
    (bounded), proving we send a curated slice rather than an ever-growing transcript.
"""

import json
import re

_ID_RE = re.compile(r"\b(S\d{2,}|SYN-\d+)\b", re.IGNORECASE)


def _ids_in(obj) -> set[str]:
    return {m.upper() for m in _ID_RE.findall(json.dumps(obj, ensure_ascii=False, default=str))}


def audit_payload(
    context: dict, active_student_id: str, config: dict, size_chars: int | None = None
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    active = active_student_id.upper()

    # 1. no other student's identifiers anywhere in the tool outputs
    foreign = {i for i in _ids_in(context.get("tool_outputs", [])) if i != active}
    if foreign:
        issues.append(f"foreign student id(s) in tool outputs: {sorted(foreign)}")

    # 2. the payload itself is scoped to the active student
    if (context.get("student_id") or "").upper() != active:
        issues.append(f"payload student_id '{context.get('student_id')}' != active '{active}'")

    # 3. size budget
    max_chars = int(config["context"]["max_chars"])
    sc = (
        size_chars
        if size_chars is not None
        else len(json.dumps(context, ensure_ascii=False, default=str))
    )
    if sc > max_chars:
        issues.append(f"context size {sc} exceeds budget {max_chars}")

    # 4. mastery slice respects top-K
    top_k = int(config["memory"]["top_k_mastery_in_context"])
    ranked = context.get("memory", {}).get("ranked_topics", [])
    if len(ranked) > top_k:
        issues.append(f"ranked_topics {len(ranked)} exceeds top_k {top_k}")

    # 5. session slice is bounded (assembler caps at 5)
    turns = context.get("memory", {}).get("session_turns", [])
    if len(turns) > 6:
        issues.append(f"session_turns {len(turns)} is unbounded")

    return (not issues, issues)


def audit_session_growth(sizes: list[int], config: dict) -> tuple[bool, list[str]]:
    """Context must stay bounded as the session grows: always within the char budget, and
    with no upward trend across turns (later turns are not systematically larger — the size
    tracks the current turn's tool outputs, not an accumulating transcript)."""
    if len(sizes) < 2:
        return True, []
    issues: list[str] = []
    max_chars = int(config["context"]["max_chars"])
    over = [s for s in sizes if s > max_chars]
    if over:
        issues.append(f"turn(s) exceeded budget {max_chars}: {over}")
    half = len(sizes) // 2
    first_avg = sum(sizes[:half]) / half
    second_avg = sum(sizes[half:]) / (len(sizes) - half)
    if second_avg > first_avg * 1.5 + 400:
        issues.append(
            f"upward size trend: first-half avg {first_avg:.0f} -> second-half avg {second_avg:.0f}"
        )
    return (not issues, issues)
