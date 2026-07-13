"""pipeline.mode == 'hybrid' — adversarial suite, proving the deterministic pre-router's
guardrail short-circuit is inherited UNCHANGED: a client that raises if ever consulted still
lets every true-positive adversarial case resolve correctly, because the LLM is never reached
for these (src/orchestrator/router.py — safety.direct_answer is returned before select_tools
is ever called). Compare directly against tests/mode_deterministic/test_deterministic_adversarial.py, which
runs the identical suite and — because both modes share the same pre-router — passes
identically.

This is the mode's one unambiguous *strength* relative to llm mode: see
tests/mode_llm/test_llm_known_failures.py (case g1) and PHASE2_REPORT.md §4.3 for the confirmed
jailbreak-refusal failure that appears only once this pre-router is removed.
"""

import pathlib

import pytest
import yaml

from conftest import RaisingClient, hybrid_orchestrator

_ADV = yaml.safe_load(
    (pathlib.Path(__file__).resolve().parents[2] / "eval" / "adversarial_queries.yaml").read_text()
)
_STUDENT = _ADV.get("student", "S123")


def _query(case):
    if "query_repeat" in case:
        ch, count = case["query_repeat"]
        return ch * int(count)
    return case["query"]


@pytest.mark.parametrize(
    "case", [c for c in _ADV["cases"] if not c.get("inject")], ids=lambda c: c["id"]
)
def test_hybrid_adversarial_suite_never_reaches_llm(monkeypatch, reset_memory, config, case):
    reset_memory(_STUDENT)
    orch = hybrid_orchestrator(monkeypatch, config, RaisingClient())
    r = orch.respond(_query(case), _STUDENT, orch.new_session())
    if "expect_intent" in case:
        assert r["intent"] in case["expect_intent"], f"{case['id']}: intent {r['intent']}"
    if case.get("forbid_tools"):
        assert r["tools_called"] == [], f"{case['id']}: unexpected tools {r['tools_called']}"
    for frag in case.get("forbid_text", []):
        assert frag.lower() not in r["answer"].lower(), f"{case['id']}: leaked '{frag}'"
