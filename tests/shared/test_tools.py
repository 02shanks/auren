"""Tool-layer tests (blueprint sec 5): schemas, structured errors, past-date filtering,
exact/semantic retrieval, no-fabrication, and safe dispatch. Mode-agnostic — every tool is a
plain ``run(ctx, **kwargs) -> dict`` function called directly via the registry's ``dispatch``,
with no LLM or routing involved, so these hold identically across deterministic, hybrid, and
llm pipeline modes."""

from src.tools.registry import dispatch, tool_names, tool_specs


def test_registry_exposes_five_tools():
    names = tool_names()
    for t in [
        "get_weak_topics",
        "get_upcoming_tests",
        "get_performance_summary",
        "recommend_study_material",
        "log_feedback",
    ]:
        assert t in names
    # every spec is a well-formed function schema
    for spec in tool_specs():
        assert spec["type"] == "function"
        assert "name" in spec["function"] and "parameters" in spec["function"]


def test_get_weak_topics(orch):
    out = dispatch("get_weak_topics", orch.ctx, {"student_id": "S123"})
    topics = [w["topic"] for w in out["weak_topics"]]
    assert "Algebra" in topics
    assert out["count"] == len(out["weak_topics"])


def test_upcoming_tests_filters_past(orch):
    out = dispatch("get_upcoming_tests", orch.ctx, {"student_id": "S123"})
    ids = [t["test_id"] for t in out["upcoming_tests"]]
    assert "T201" not in ids  # 2026-04-14 is in the past relative to 2026-07-05
    assert any(f["test_id"] == "T201" and f["reason"] == "past" for f in out["filtered_out"])


def test_upcoming_tests_future_kept(orch):
    out = dispatch("get_upcoming_tests", orch.ctx, {"student_id": "SYN-17"})
    ids = [t["test_id"] for t in out["upcoming_tests"]]
    assert "T312" in ids  # 2026-07-15 is in the future


def test_performance_summary(orch):
    out = dispatch("get_performance_summary", orch.ctx, {"student_id": "S123"})
    subjects = {s["subject"] for s in out["subjects"]}
    assert "Mathematics" in subjects


def test_recommend_exact_match(orch):
    out = dispatch("recommend_study_material", orch.ctx, {"topic": "Algebra", "top_k": 1})
    assert out["recommendations"], "expected a recommendation for Algebra"
    top = out["recommendations"][0]
    assert top["material_id"] == "M101"
    assert top["match_type"] == "exact"


def test_recommend_never_fabricates(orch):
    out = dispatch(
        "recommend_study_material", orch.ctx, {"topic": "Zxqwv Imaginary Topic", "top_k": 3}
    )
    assert out["recommendations"] == []  # honest no-match, nothing invented


def test_log_feedback_updates_and_persists(orch, reset_memory):
    reset_memory("S123")
    dispatch("get_weak_topics", orch.ctx, {"student_id": "S123"})  # establish baseline
    out = dispatch(
        "log_feedback", orch.ctx, {"student_id": "S123", "topic": "Algebra", "signal": "helped"}
    )
    assert out["updated_priority_score"] is not None
    assert out["previous_priority_score"] is None or (
        out["updated_priority_score"] <= out["previous_priority_score"]
    )


def test_log_feedback_rejects_bad_signal(orch, reset_memory):
    reset_memory("S123")
    out = dispatch(
        "log_feedback", orch.ctx, {"student_id": "S123", "topic": "Algebra", "signal": "meh"}
    )
    assert out["error"] == "invalid_signal"


def test_tool_absence_is_structured_not_exception(orch):
    out = dispatch("get_weak_topics", orch.ctx, {"student_id": "S999-missing"})
    assert out["error"] == "student_not_found"


def test_dispatch_unknown_tool(orch):
    out = dispatch("no_such_tool", orch.ctx, {"student_id": "S123"})
    assert out["error"] == "unknown_tool"


def test_dispatch_drops_unknown_args(orch):
    # an unexpected kwarg must be filtered, not crash the tool
    out = dispatch("get_weak_topics", orch.ctx, {"student_id": "S123", "injected": "DROP TABLE"})
    assert "weak_topics" in out
    assert out.get("error") != "tool_exception"


def test_duplicate_student_raises_integrity_in_tool(orch):
    # SYN-05 collides with SYN-18's internal id -> tool reports a structured integrity error
    out = dispatch("get_weak_topics", orch.ctx, {"student_id": "SYN-05"})
    assert out["error"] == "data_integrity_error"
