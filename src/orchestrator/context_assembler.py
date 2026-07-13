"""Context assembly with an explicit contract.

Every field in the assembled payload is attributable to (a) a tool called THIS turn,
(b) a size-bounded memory slice, or (c) the fixed system prompt. Isolation is
structural: this function only ever receives the active student's mastery/persona/
session, so another student's data is never in scope to leak. A hard character
budget is enforced by trimming (never by reaching for a bigger model).
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

log = get_logger("context")

SESSION_TURN_CAP = 5  # episodic slice is the current session only, and bounded


class ContextPayload(BaseModel):
    student_id: str
    query: str
    intent: str
    focus_topic: str | None = None
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    ranked_topics: list[tuple[str, float]] = Field(default_factory=list)
    session_turns: list[dict[str, Any]] = Field(default_factory=list)
    persona: dict[str, Any] = Field(default_factory=dict)
    size_chars: int = 0
    approx_tokens: int = 0
    oversized: bool = False

    def to_llm_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "intent": self.intent,
            "focus_topic": self.focus_topic,
            "tool_outputs": self.tool_outputs,
            "memory": {
                "ranked_topics": [[t, s] for t, s in self.ranked_topics],
                "session_turns": self.session_turns,
            },
            "persona": self.persona,
        }

    def log_summary(self) -> dict[str, Any]:
        """Field-level summary + hash for the sec 14.4 context audit (never the raw dump)."""
        blob = json.dumps(self.to_llm_dict(), ensure_ascii=False, sort_keys=True)
        return {
            "student_id": self.student_id,
            "intent": self.intent,
            "tools_in_context": [o["tool"] for o in self.tool_outputs],
            "ranked_topics_count": len(self.ranked_topics),
            "session_turns_count": len(self.session_turns),
            "persona_included": bool(self.persona),
            "size_chars": self.size_chars,
            "approx_tokens": self.approx_tokens,
            "oversized": self.oversized,
            "payload_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        }


def _measure(payload: ContextPayload, cpt: int) -> None:
    blob = json.dumps(payload.to_llm_dict(), ensure_ascii=False)
    payload.size_chars = len(blob)
    payload.approx_tokens = payload.size_chars // max(1, cpt)


def assemble_context(
    query: str,
    intent: str,
    tool_outputs: list[dict[str, Any]],
    ranked_topics: list[tuple[str, float]],
    session_turns: list[dict[str, Any]],
    persona: dict[str, Any],
    config: dict,
    focus_topic: str | None = None,
) -> ContextPayload:
    mem = config.get("memory", {})
    ctxc = config.get("context", {})
    k = int(mem.get("top_k_mastery_in_context", 3))
    min_evidence = int(mem.get("persona_min_evidence", 2))
    max_chars = int(ctxc.get("max_chars", 4800))
    cpt = int(ctxc.get("approx_chars_per_token", 4))

    # persona only if it clears the evidence bar (drift guard)
    persona_slice: dict[str, Any] = {}
    if (
        persona.get("preferred_material_type")
        and int(persona.get("evidence_count", 0)) >= min_evidence
    ):
        persona_slice = {
            "preferred_material_type": persona["preferred_material_type"],
            "evidence_count": persona["evidence_count"],
            "confidence": persona.get("confidence"),
        }

    payload = ContextPayload(
        student_id=session_turns[0]["student_id"] if session_turns else "",
        query=query,
        intent=intent,
        focus_topic=focus_topic,
        tool_outputs=tool_outputs,
        ranked_topics=[(t, float(s)) for t, s in ranked_topics[:k]],
        session_turns=[
            {"query": t.get("query", "")[:200], "intent": t.get("intent")}
            for t in session_turns[-SESSION_TURN_CAP:]
        ],
        persona=persona_slice,
    )
    # student_id is authoritative from the caller, not the session list:
    payload.student_id = _resolve_sid(session_turns, tool_outputs)

    _measure(payload, cpt)
    # enforce the hard budget by trimming, in order of least value first
    if payload.size_chars > max_chars:
        payload.session_turns = []
        _measure(payload, cpt)
    if payload.size_chars > max_chars:
        payload.ranked_topics = payload.ranked_topics[:1]
        _measure(payload, cpt)
    if payload.size_chars > max_chars:
        for item in payload.tool_outputs:
            out = item.get("output", {})
            for key, val in list(out.items()):
                if isinstance(val, list) and len(val) > 3:
                    out[key] = val[:3]
        _measure(payload, cpt)
        payload.oversized = payload.size_chars > max_chars
        log.warning(
            "context still %d chars after trimming (budget %d)", payload.size_chars, max_chars
        )
    return payload


def _resolve_sid(session_turns: list[dict[str, Any]], tool_outputs: list[dict[str, Any]]) -> str:
    for o in tool_outputs:
        sid = o.get("output", {}).get("student_id")
        if sid:
            return sid
    if session_turns:
        return session_turns[0].get("student_id", "")
    return ""
