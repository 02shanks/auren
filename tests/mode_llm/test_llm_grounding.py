"""pipeline.mode == 'llm' — post-generation grounding verification (src/llm/agentic.py's
``AgenticExecutor.ground``), the one structural safety net this mode keeps in place of the
deterministic router. Covers both configured policies: ``enforce`` (critique-regen, then a
grounded-by-construction fallback) and ``report`` (flag without replacing).

Note the real, confirmed gap this mechanism has in practice: it only verifies ids/dates/numbers
against the evidence pool (src/orchestrator/grounding.py), not prose claims like a fabricated
test *name* or material *title* — see tests/mode_llm/test_llm_known_failures.py (cases a4, b2) and
PHASE2_REPORT.md §3 for the two hallucinations that passed grounding_ok=true regardless.
"""

from src.llm.agentic import AgenticExecutor, AgenticResult
from src.utils.config import load_config

from conftest import FakeRawClient


def _executor(policy, script):
    config = load_config()
    config.setdefault("pipeline", {}).setdefault("llm", {})["grounding"] = policy
    fake = FakeRawClient(script)
    return AgenticExecutor(fake, config, lambda c, s: {}, tools=[]), fake


def test_grounding_enforce_falls_back_when_regen_fails():
    ex, _ = _executor("enforce", [{"content": "Score 99% on M999 guaranteed."}])
    res = AgenticResult(answer="You scored 87% on test T777.", intent="performance_query")
    ctx = {"student_id": "S123", "intent": "performance_query", "tool_outputs": [], "memory": {}}
    ex.ground(res, ctx)
    assert res.grounded_fallback is True
    assert res.grounding_ok is True  # fallback is grounded by construction
    assert "87" not in res.answer and "T777" not in res.answer


def test_grounding_report_flags_without_replacing():
    ex, _ = _executor("report", [{"content": "Still says 87% on T777."}])
    res = AgenticResult(answer="You scored 87% on test T777.", intent="performance_query")
    ctx = {"student_id": "S123", "intent": "performance_query", "tool_outputs": [], "memory": {}}
    ex.ground(res, ctx)
    assert res.grounding_ok is False
    assert "87%" in res.answer  # answer kept, flagged instead of replaced
