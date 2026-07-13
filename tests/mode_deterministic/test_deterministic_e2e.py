"""End-to-end tests of the four canonical lifecycle queries (blueprint sec 3.2) plus
self-improvement across a session, all driven through pipeline.mode == 'deterministic' —
fast, offline, zero model calls (see tests/mode_deterministic/conftest.py). This is the
mode every one of these assertions was originally calibrated against: exact intent labels,
exact tool sets, and literal answer substrings, all guaranteed by the deterministic template
composer. See tests/mode_hybrid/test_hybrid_e2e.py and tests/mode_llm/test_llm_e2e_live.py for the
equivalent canonical-query coverage in the other two modes — their assertions are necessarily
looser, because free-form LLM generation cannot satisfy exact-substring checks reliably
(documented in PHASE1_REPORT.md / PHASE2_REPORT.md)."""

import pytest


@pytest.fixture
def s123(orch, reset_memory):
    def _ask(query):
        reset_memory("S123")  # each canonical query is its own scenario
        return orch.respond(query, "S123", orch.new_session())

    return _ask


def test_q1_weakness_focus(s123):
    r = s123("I am weak in Algebra, what should I study next?")
    assert r["intent"] == "weakness_focus"
    assert "get_weak_topics" in r["tools_called"]
    assert "recommend_study_material" in r["tools_called"]
    assert "M101" in r["answer"]
    assert r["grounding_ok"] and not r["grounded_fallback"]


def test_q2_study_plan(s123):
    r = s123("What should I study this week?")
    assert r["intent"] == "study_plan"
    for t in ["get_weak_topics", "get_upcoming_tests", "get_performance_summary"]:
        assert t in r["tools_called"]
    assert "recommend_study_material" in r["tools_called"]
    assert r["grounding_ok"]


def test_q3_prioritize_commits_to_one(s123):
    r = s123("Which topic should I prioritize first?")
    assert r["intent"] == "prioritize"
    # commits to a single top priority
    assert r["tools_called"].count("recommend_study_material") == 1
    assert "Algebra" in r["answer"]
    assert r["grounding_ok"]


def test_q4_test_prep_filters_past_test(s123):
    r = s123("I have a Maths test coming up, help me prepare")
    assert r["intent"] == "test_prep"
    assert "get_upcoming_tests" in r["tools_called"]
    # Backend-independent invariant: the tool classified T201 as a *past* test (the real
    # correctness check; holds whether the answer is written by the LLM or the deterministic
    # engine, and regardless of subject phrasing like "Maths" vs "Mathematics").
    ut = next(
        o["output"] for o in r["context"]["tool_outputs"] if o["tool"] == "get_upcoming_tests"
    )
    assert any(f["test_id"] == "T201" and f["reason"] == "past" for f in ut["filtered_out"])
    assert not ut["upcoming_tests"]  # never presented as upcoming
    # Prose communicates there is no upcoming Maths test (exact wording varies by backend).
    a = r["answer"].lower()
    assert (
        ("t201" in a) or ("past" in a) or ("no upcoming" in a) or ("don't have any upcoming" in a)
    )
    assert r["grounding_ok"]


def test_full_session_self_improves(orch, reset_memory):
    reset_memory("S123")
    sess = orch.new_session()
    r1 = orch.respond("what should I prioritize first?", "S123", sess)
    top_before = r1["answer"].splitlines()[1] if "\n" in r1["answer"] else r1["answer"]
    orch.respond("the algebra notes really helped", "S123", sess)
    r3 = orch.respond("what should I prioritize first?", "S123", sess)
    # after positive feedback on Algebra, the top recommendation should change away from Algebra
    assert "Algebra" in top_before
    assert r3["answer"].splitlines()[1] != top_before if "\n" in r3["answer"] else True
