"""Context-payload mechanics (blueprint sec 8.4): cap/budget/trim behavior of
``assemble_context`` and the standalone grounding-checker regex, exercised as pure functions
with no orchestrator turn and no backend involved — mode-agnostic by construction, since every
pipeline mode shares the same context assembler and grounding checker."""

from src.orchestrator.context_assembler import assemble_context
from src.orchestrator.grounding import check_grounding


def _turns(n, sid="S123"):
    return [{"student_id": sid, "query": f"q{i}", "intent": "study_plan"} for i in range(n)]


def test_ranked_topics_capped_to_top_k(orch):
    ranked = [(f"Topic{i}", 0.9 - 0.02 * i) for i in range(12)]
    payload = assemble_context(
        "q",
        "study_plan",
        [{"tool": "get_weak_topics", "output": {"weak_topics": []}}],
        ranked,
        _turns(1),
        {},
        orch.config,
    )
    assert len(payload.ranked_topics) <= orch.config["memory"]["top_k_mastery_in_context"]


def test_session_turns_capped(orch):
    payload = assemble_context(
        "q", "study_plan", [], [("Algebra", 0.6)], _turns(8), {}, orch.config
    )
    assert len(payload.session_turns) <= 5


def test_oversized_context_trimmed_to_budget(orch):
    big = [
        {
            "tool": "get_weak_topics",
            "output": {
                "weak_topics": [{"topic": "X" * 300, "subject": "Y" * 300} for _ in range(60)]
            },
        }
    ]
    payload = assemble_context(
        "q", "study_plan", big, [("Algebra", 0.6)], _turns(1), {}, orch.config
    )
    assert payload.size_chars <= orch.config["context"]["max_chars"] or payload.oversized


def test_grounding_flags_foreign_id():
    ctx = {
        "student_id": "S123",
        "intent": "study_plan",
        "tool_outputs": [
            {"tool": "get_weak_topics", "output": {"weak_topics": [{"topic": "Algebra"}]}}
        ],
        "memory": {"ranked_topics": [["Algebra", 0.6]], "session_turns": []},
        "persona": {},
    }
    ok, problems = check_grounding("Study Algebra. Also, S777 is failing everything.", ctx)
    assert ok is False and problems
