"""Reflection cycle

At session end we (1) summarize the session's episodic entries into 1-2 sentences
and (2) update persona overrides *only* with >=2 corroborating events (the drift
guard). Persona evidence is tallied from the durable episodic log, so it genuinely
accumulates across sessions rather than flipping on a single instance.
"""

from typing import Any

from src.memory.store import MemoryStore
from src.utils.logging_config import get_logger

log = get_logger("reflection")

POSITIVE = {"helped", "positive", "up", "good"}


def summarize_session(session_episodes: list[dict[str, Any]]) -> str:
    if not session_episodes:
        return "No queries this session."
    n = len(session_episodes)
    topics: list[str] = []
    for e in session_episodes:
        for t in e.get("topics") or []:
            if t not in topics:
                topics.append(t)
    fb = [e for e in session_episodes if e.get("intent") == "feedback"]
    plural = "y" if n == 1 else "ies"
    parts = [f"Session covered {n} quer{plural}."]
    if topics:
        parts.append("Focus topics: " + ", ".join(topics[:3]) + ".")
    if fb:
        helped = sum(1 for e in fb if e.get("signal") in POSITIVE)
        parts.append(f"Logged {len(fb)} feedback event(s), {helped} positive.")
    return " ".join(parts)


def update_persona(store: MemoryStore, config: dict) -> dict[str, Any]:
    min_evidence = int(config.get("memory", {}).get("persona_min_evidence", 2))
    counts: dict[str, int] = {}
    for e in store.read_episodes():
        if e.get("intent") == "feedback" and e.get("signal") in POSITIVE and e.get("material_type"):
            mt = str(e["material_type"])
            counts[mt] = counts.get(mt, 0) + 1
    persona = store.load_persona()
    if not counts:
        return persona
    persona["_type_counts"] = counts
    top_type, top_n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if top_n >= min_evidence:
        persona["preferred_material_type"] = top_type
        persona["evidence_count"] = top_n
        persona["confidence"] = round(min(0.95, 0.3 + 0.15 * top_n), 2)
        log.info("persona updated: preferred_material_type=%s (evidence=%d)", top_type, top_n)
    store.save_persona(persona)
    return persona


def run_reflection(store: MemoryStore, session_id: str, config: dict) -> tuple[str, dict[str, Any]]:
    episodes = store.read_session_episodes(session_id)
    summary = summarize_session(episodes)
    persona = update_persona(store, config)
    store.append_episode(
        {"session_id": session_id, "intent": "reflection", "response_summary": summary}
    )
    return summary, persona
