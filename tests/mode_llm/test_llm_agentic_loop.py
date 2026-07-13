"""pipeline.mode == 'llm' — the end-to-end agentic tool-calling loop (src/llm/agentic.py),
offline via a scripted FakeRawClient (see conftest.py). Covers: multi-round tool execution,
structural validation surviving a hallucinated tool name and a foreign student_id, that the
deterministic keyword router genuinely never runs in this mode (paraphrases it would decline
reach the model instead), that a feedback-driving tool call still maps to the 'feedback'
intent so persona/reflection keeps working, bounded session history across turns, and the
fail-fast requirement for a raw-chat-capable backend.
"""

import datetime as dt

import pytest

from conftest import llm_orchestrator
from src.orchestrator.pipeline import Orchestrator

TODAY = dt.date(2026, 7, 5)


def test_llm_mode_tool_loop_and_validation(monkeypatch, reset_memory, config):
    reset_memory("S123")
    script = [
        {  # round 1: model calls a tool — with a foreign student id that must be rewritten
            "tool_calls": [
                {"name": "get_weak_topics", "arguments": {"student_id": "S999"}, "id": None},
                {"name": "made_up_tool", "arguments": {}, "id": None},
            ]
        },
        {"content": "Your weak topics are Algebra, Quadratic Equations, and Light."},
    ]
    orch, fake = llm_orchestrator(monkeypatch, config, script)
    r = orch.respond("what am I bad at?", "S123", orch.new_session())

    assert r["intent"] == "weakness_list"
    assert r["tools_called"] == ["get_weak_topics"]  # hallucinated tool dropped
    # the executed tool must have run for the ACTIVE student despite the foreign id
    wt = next(o for o in r["context"]["tool_outputs"] if o["tool"] == "get_weak_topics")
    assert wt["output"]["student_id"] == "S123"
    assert "Algebra" in r["answer"]
    assert r["grounding_ok"] is True


def test_llm_mode_no_deterministic_shortcircuit(monkeypatch, reset_memory, config):
    """Phrasings the keyword router declines as off_topic must reach the model — see
    PHASE1_REPORT.md §3.1 for the same query failing in deterministic/hybrid mode."""
    reset_memory("S123")
    script = [{"content": "Let's work on your weak areas together."}]
    orch, fake = llm_orchestrator(monkeypatch, config, script)
    r = orch.respond("I keep messing up trigonometry sums", "S123", orch.new_session())
    assert len(fake.calls) == 1  # the model was consulted, not short-circuited
    assert "weak areas" in r["answer"]
    assert r["intent"] == "llm_agentic"


def test_llm_mode_feedback_intent_keeps_persona_channel(monkeypatch, reset_memory, config):
    reset_memory("S123")
    script = [
        {
            "tool_calls": [
                {
                    "name": "log_feedback",
                    "arguments": {"student_id": "S123", "topic": "Algebra", "signal": "helped"},
                    "id": None,
                }
            ]
        },
        {"content": "Recorded — glad Algebra practice helped."},
    ]
    orch, _ = llm_orchestrator(monkeypatch, config, script)
    r = orch.respond("the algebra notes really helped", "S123", orch.new_session())
    assert r["intent"] == "feedback"  # reflection/persona learning keys on this label
    assert "log_feedback" in r["tools_called"]


def test_llm_mode_session_history_carried(monkeypatch, reset_memory, config):
    reset_memory("S123")
    script = [
        {"content": "Focus on Algebra first."},
        {"content": "After Algebra, take up Quadratic Equations."},
    ]
    orch, fake = llm_orchestrator(monkeypatch, config, script)
    sess = orch.new_session()
    orch.respond("what first?", "S123", sess)
    orch.respond("and after that?", "S123", sess)
    # second call's prompt must contain the first turn as history
    second_messages = fake.calls[1][0]
    joined = " | ".join(str(m.get("content", "")) for m in second_messages)
    assert "what first?" in joined
    assert "Focus on Algebra first." in joined


def test_llm_mode_requires_raw_capable_backend(config):
    config.setdefault("llm", {})["backend"] = "deterministic"
    with pytest.raises(RuntimeError, match="raw-chat capable"):
        Orchestrator(config, dataset="all", today=TODAY)
