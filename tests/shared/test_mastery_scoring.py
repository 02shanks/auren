"""Mastery-scoring tests (blueprint sec 7.2): the exact priority formula, weak>strong ordering,
the urgency term, and feedback measurably changing scores and ranking for multiple students.
Mode-agnostic — ``MasteryEngine`` is called directly and never touches an LLM client or the
router, so the scoring formula holds identically across all three pipeline modes; it is the
*callers* of ``log_feedback``/``apply_feedback`` (deterministic's keyword extraction vs. an
LLM's tool-call arguments) that differ by mode — see tests/mode_hybrid/test_hybrid_known_failures.py
and tests/mode_llm/test_llm_known_failures.py for the one confirmed cross-mode bug in that layer
(a material-id-referenced feedback event landing against the wrong topic key)."""

import datetime as dt

import pytest

from src.memory.mastery import MasteryEngine

TODAY = dt.date(2026, 7, 5)


def _mastery(orch, sid, existing=None):
    rec = orch.repo.get_student(sid)
    return orch.mastery.recompute(rec, existing or {}, today=TODAY)


def test_weak_topic_priority_formula(orch):
    # Algebra is weak; its only test (T201) is in the past -> no urgency; fresh -> staleness maxed.
    # 0.40*weakness(1) + 0.30*urgency(0) + 0.20*staleness(1) - 0.20*decay(0) = 0.60
    m = _mastery(orch, "S123")
    assert abs(m["Algebra"]["priority_score"] - 0.60) < 0.02


def test_weak_outranks_strong(orch):
    m = _mastery(orch, "S123")
    scores = dict(MasteryEngine.ranked_topics(m))
    weak_min = min(scores[t] for t in ["Algebra", "Quadratic Equations"])
    strong_max = max(scores.get(t, 0.0) for t in ["Linear Equations", "Chemical Reactions"])
    assert weak_min > strong_max


def test_urgency_raises_priority(orch):
    # SYN-04's Algebra is tested in ~13 days -> urgency pushes it above the no-urgency baseline 0.60
    m = _mastery(orch, "SYN-04")
    assert m["Algebra"]["priority_score"] > 0.60


@pytest.mark.parametrize("sid", ["S123", "SYN-04", "SYN-17"])
def test_positive_feedback_lowers_priority(orch, sid):
    base = _mastery(orch, sid)
    before = base["Algebra"]["priority_score"]
    updated = orch.mastery.apply_feedback(base, "Algebra", "helped", today=TODAY)
    after = orch.mastery.recompute(orch.repo.get_student(sid), updated, today=TODAY)
    assert after["Algebra"]["priority_score"] < before


def test_feedback_changes_top_ranking(orch):
    base = _mastery(orch, "S123")
    ranked_before = MasteryEngine.ranked_topics(base)
    top_before, top_before_score = ranked_before[0]
    updated = orch.mastery.apply_feedback(base, top_before, "helped", today=TODAY)
    after = orch.mastery.recompute(orch.repo.get_student("S123"), updated, today=TODAY)
    ranked_after = MasteryEngine.ranked_topics(after)
    # either a different topic is now on top, or the same topic's score fell
    assert ranked_after[0][0] != top_before or ranked_after[0][1] < top_before_score
