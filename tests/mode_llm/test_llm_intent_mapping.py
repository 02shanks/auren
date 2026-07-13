"""pipeline.mode == 'llm' — mapping executed tool calls to the episodic-log intent vocabulary
(src/llm/agentic.py's ``_intent_from_tools``). Without a keyword classifier assigning intent,
this mapping is what keeps memory features that key on specific intent strings (persona
learning on 'feedback', the composer's per-intent branches) working identically to the other
two modes."""

from src.llm.agentic import _intent_from_tools
from src.llm.base import ToolCall


def _tc(name):
    return ToolCall(name, {})


def test_intent_from_tools_mapping():
    assert _intent_from_tools([_tc("log_feedback"), _tc("get_weak_topics")]) == "feedback"
    assert _intent_from_tools([_tc("get_upcoming_tests"), _tc("get_weak_topics")]) == "test_prep"
    assert _intent_from_tools([_tc("get_performance_summary")]) == "performance_query"
    assert _intent_from_tools([_tc("get_weak_topics")]) == "weakness_list"
    assert _intent_from_tools([_tc("recommend_study_material")]) == "weakness_focus"
    assert _intent_from_tools([]) == "llm_agentic"
