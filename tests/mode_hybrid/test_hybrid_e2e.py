"""pipeline.mode == 'hybrid' — end-to-end wiring proof using a scripted fake LLM client (the
old-style select_tools/generate protocol hybrid mode actually calls — NOT the raw chat_raw
loop llm mode uses; see src/orchestrator/router.py). Confirms the canonical queries still
resolve correctly when a (fake) model's tool choices are unioned on top of the deterministic
classifier's, and that its free-form generate() output replaces the deterministic template.

For a REAL Ollama-backed smoke test of this same mode, see test_hybrid_e2e_live.py (skips itself if
no Ollama server is reachable). For the confirmed hallucinations this mode produced in the
Phase 1 evaluation, see test_hybrid_known_failures.py.
"""

from src.llm.base import ToolCall

from conftest import FakeSelectGenerateClient, hybrid_orchestrator


def test_hybrid_canonical_query_uses_deterministic_tools_and_llm_prose(
    monkeypatch, reset_memory, config
):
    reset_memory("S123")
    fake = FakeSelectGenerateClient("Start with Algebra — see M101.")
    orch = hybrid_orchestrator(monkeypatch, config, fake)
    r = orch.respond("I am weak in Algebra, what should I study next?", "S123", orch.new_session())

    # the deterministic pre-router still supplies the tool plan (hybrid's design: LLM choices
    # are UNIONED on top, and an empty LLM selection changes nothing about which tools run)
    assert "get_weak_topics" in r["tools_called"]
    assert "recommend_study_material" in r["tools_called"]
    assert r["intent"] == "weakness_focus"
    # the (fake) LLM's generate() output is what's returned, not the deterministic template
    assert r["answer"] == fake.generate_text
    assert len(fake.generate_calls) == 1


def test_hybrid_unions_llm_tool_choice_on_top_of_forced_set(monkeypatch, reset_memory, config):
    reset_memory("S123")
    fake = FakeSelectGenerateClient(
        extra_tool_calls=[ToolCall("get_performance_summary", {"student_id": "S123"})]
    )
    orch = hybrid_orchestrator(monkeypatch, config, fake)
    r = orch.respond("What should I study this week?", "S123", orch.new_session())

    # the LLM's extra tool call is unioned on top of the deterministic study_plan tool set
    assert "get_performance_summary" in r["tools_called"]
    assert "get_weak_topics" in r["tools_called"]


## See test_hybrid_adversarial.py for the full proof (parametrized over every adversarial_queries.yaml
## case) that the deterministic pre-router's guardrail short-circuit means the LLM is never
## even consulted for an unambiguous jailbreak in this mode.
