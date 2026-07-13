"""pipeline.mode == 'hybrid' — documented regressions from the Phase 1 evaluation
(PHASE1_REPORT.md §3.1, §3.6), replayed through a scripted fake client so the pipeline's
current (buggy) handling of each transcript is pinned down and won't silently change or
silently reappear unnoticed.

Each test replays the EXACT recorded model output from eval/phase1_results_hybrid.jsonl for
the named case id — it does not re-verify that a live qwen3:8b still says the same thing (see
`uv run python -m eval.phase1_runner --mode hybrid` for that). What it verifies is that the
pipeline's post-generation grounding check does not catch this class of hallucination, which is
the actual, still-open finding (PHASE2_REPORT.md §3: grounding only checks ids/dates/numbers,
never prose claims like an exam score attribution or a fabricated test's relevance).

These tests are expected to keep FAILING their "would be nice" assertion (marked xfail) until
PHASE2_REPORT.md §8's grounding-extension recommendation is implemented — at which point the
xfail marker should be removed and the test re-asserts the CORRECT behavior instead.
"""

import pytest

from src.llm.base import ToolCall

from conftest import FakeSelectGenerateClient, hybrid_orchestrator


def test_b11_score_aggregate_misreported_as_exam_score(monkeypatch, reset_memory, config):
    """Case b11: 'What did I score in my last science exam?' — the student has no per-exam
    record, only a per-subject aggregate (Science 63%, data/performance_history.json). The
    recorded hybrid answer stated this aggregate AS an exam score, a specific false claim
    stated as fact rather than an honest 'no exam-level record' answer."""
    reset_memory("S123")
    recorded_answer = (
        'You scored 63% in your last science exam. Focus on Algebra first — start with '
        '"Algebra Basics Revision Notes" (M101).'
    )
    fake = FakeSelectGenerateClient(recorded_answer)
    orch = hybrid_orchestrator(monkeypatch, config, fake)
    r = orch.respond("What did I score in my last science exam?", "S123", orch.new_session())

    assert r["answer"] == recorded_answer
    # documents that grounding does NOT catch this: 63 is a real, evidence-backed number (the
    # Science subject aggregate), so the regex-based numeric check has nothing to flag even
    # though the claim it's attached to ("last science exam") is unsupported by any tool output
    assert r["grounding_ok"] is True


@pytest.mark.xfail(
    reason=(
        "known bug (PHASE1_REPORT.md §3.1): hybrid asserts an upcoming Science test exists "
        "when data/upcoming_tests.json has none for S123; grounding does not catch a fabricated "
        "test's *relevance* claim, only ids/dates/numbers. Fix per PHASE2_REPORT.md §8, then "
        "flip this assertion to require an honest 'no Science test on record' answer."
    ),
    strict=True,
)
def test_b15_fabricates_nonexistent_science_test(monkeypatch, reset_memory, config):
    """Case b15: 'Explain what I should focus on before my Science test' — S123 has NO Science
    test on record at all (data/upcoming_tests.json has one Mathematics test). The recorded
    hybrid answer asserted the topic 'is a key topic for your Science test', inventing a test
    that does not exist."""
    reset_memory("S123")
    recorded_answer = (
        "Focus on Light - Reflection and Refraction first, as it is a key topic for your "
        "Science test. Review the relevant material thoroughly. Then prioritize Algebra and "
        "Quadratic Equations, as these are also important areas to master. Make sure you "
        "understand the core concepts in each of these topics."
    )
    fake = FakeSelectGenerateClient(recorded_answer)
    orch = hybrid_orchestrator(monkeypatch, config, fake)
    r = orch.respond(
        "Explain what I should focus on before my Science test", "S123", orch.new_session()
    )
    # the CORRECT behavior (what this test will require once fixed): an honest answer must not
    # claim a Science test exists
    assert "science test" not in r["answer"].lower() or "no" in r["answer"].lower()


@pytest.mark.xfail(
    reason=(
        "known bug (PHASE1_REPORT.md §3.6): log_feedback has no material_id->topic resolution "
        "of its own (src/tools/log_feedback.py); when the LLM's own tool call wins the union "
        "(router.py:39-41, since both propose the same tool name), feedback for M103 (real "
        "topic: Quadratic Equations) is logged against the wrong topic 'Algebra Basics'. Fix "
        "per PHASE2_REPORT.md §8 item 4 (expose a shared material_id->topic lookup), then flip "
        "this assertion to require the correct topic."
    ),
    strict=True,
)
def test_e3_material_id_feedback_logged_against_wrong_topic(monkeypatch, reset_memory, config):
    """Case e3: 'M103 was a waste of time' — M103's real topic is Quadratic Equations
    (data/study_materials.json). The deterministic engine resolves this correctly (see
    tests/mode_deterministic/test_deterministic_e2e.py's equivalent path) by looking up material_id->topic
    itself before proposing its log_feedback call (src/llm/deterministic.py:356-362).

    In the recorded hybrid run, the answer text ("recorded that Algebra Basics wasn't
    helpful") proves the LLM proposed its OWN log_feedback call with the wrong topic — "Algebra
    Basics" is not a real topic at all, but a fragment of M101's title ("Algebra Basics
    Revision Notes"), i.e. the model confused M103 with an unrelated material. Only the exact
    wrong-topic string the model passed wasn't logged (eval/phase1_results_hybrid.jsonl records
    tool names + the final answer, not raw tool-call arguments), so this test reconstructs the
    competing tool call from that evidence to reproduce why the union (router.py:39-41 — the
    deterministic safety-net's correctly-resolved call is dropped once the LLM proposes a
    same-named call) picks the wrong one, and asserts what the topic *should* resolve to."""
    reset_memory("S123")
    fake = FakeSelectGenerateClient(
        generate_text="Got it — recorded that Algebra Basics wasn't helpful. Its priority is now 0.20.",
        extra_tool_calls=[
            ToolCall(
                "log_feedback",
                {
                    "student_id": "S123",
                    "topic": "Algebra Basics",  # wrong: M103's real topic is Quadratic Equations
                    "signal": "not_helped",
                    "material_id": "M103",
                },
            )
        ],
    )
    orch = hybrid_orchestrator(monkeypatch, config, fake)
    r = orch.respond("M103 was a waste of time", "S123", orch.new_session())
    # the CORRECT behavior: feedback must land on Quadratic Equations, M103's real topic
    assert "quadratic" in r["answer"].lower()
