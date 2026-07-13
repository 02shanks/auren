"""Auren evaluation harness (blueprint sec 14).

Runs the sample, adversarial, and golden datasets through the real orchestrator and prints
a scorecard across five dimensions:

    tool-call accuracy | groundedness | context audit | adversarial robustness | self-improvement

Every case runs on a fresh memory state with the run date pinned, so results are
deterministic. Exits non-zero if any dimension is below 100% (fail-fast for CI).

Run: ``uv run python -m eval.run_eval``
"""

import datetime as dt
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for eval.* imports

import yaml

from eval.context_audit import audit_payload, audit_session_growth
from src.orchestrator.pipeline import Orchestrator
from src.retrieval.indexer import build_index, get_retriever
from src.utils.config import load_config, repo_path
from src.utils.data_loader import load_injection_materials

EVAL_DIR = Path(__file__).resolve().parent
TODAY = dt.date(2026, 7, 5)


def _load(name: str) -> dict:
    return yaml.safe_load((EVAL_DIR / name).read_text(encoding="utf-8"))


def _reset(student_id: str, config: dict) -> None:
    d = repo_path(config["memory"]["root"]) / student_id
    if d.exists():
        shutil.rmtree(d)


def _query_of(case: dict) -> str:
    if "query_repeat" in case:
        ch, count = case["query_repeat"]
        return ch * int(count)
    return case["query"]


def _fresh(config: dict, dataset: str = "all") -> Orchestrator:
    return Orchestrator(config, dataset=dataset, today=TODAY)


def _inject(orch: Orchestrator, config: dict) -> None:
    """Add the poisoned material fixtures to the retriever's index (defense test)."""
    inj = load_injection_materials()
    have = {m.material_id for m in orch.repo.materials_list}
    orch.repo.materials_list.extend(m for m in inj if m.material_id not in have)
    build_index(orch.repo.materials(), config)
    orch.retriever = get_retriever(config, "all", materials=orch.repo.materials())
    orch.ctx.retriever = orch.retriever


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_tool_calls(config: dict) -> tuple[int, int, list[str]]:
    data = _load("sample_queries.yaml")
    passed, fails = 0, []
    orch = _fresh(config)
    for case in data["cases"]:
        _reset(case["student"], config)
        r = orch.respond(_query_of(case), case["student"], orch.new_session())
        problems = []
        if r["intent"] not in case.get("expect_intent", [r["intent"]]):
            problems.append(f"intent {r['intent']} not in {case['expect_intent']}")
        for t in case.get("expect_tools_include", []):
            if t not in r["tools_called"]:
                problems.append(f"missing tool {t}")
        for frag in case.get("expect_answer_contains", []):
            if frag.lower() not in r["answer"].lower():
                problems.append(f"answer missing '{frag}'")
        if problems:
            fails.append(f"[{case['id']}] " + "; ".join(problems))
        else:
            passed += 1
    return passed, len(data["cases"]), fails


def stage_groundedness(config: dict) -> tuple[int, int, list[str]]:
    sample = _load("sample_queries.yaml")["cases"]
    golden = _load("golden_dataset.yaml").get("grounding", [])
    cases = [(c["id"], c["student"], _query_of(c)) for c in sample] + [
        (c["id"], c["student"], c["query"]) for c in golden
    ]
    passed, fails = 0, []
    orch = _fresh(config)
    for cid, sid, q in cases:
        _reset(sid, config)
        r = orch.respond(q, sid, orch.new_session())
        if r["grounding_ok"]:
            passed += 1
        else:
            fails.append(f"[{cid}] answer not grounded: {r['answer'][:80]}")
    return passed, len(cases), fails


def stage_context_audit(config: dict) -> tuple[int, int, list[str]]:
    data = _load("sample_queries.yaml")
    passed, fails = 0, []
    orch = _fresh(config)
    for case in data["cases"]:
        _reset(case["student"], config)
        r = orch.respond(_query_of(case), case["student"], orch.new_session())
        ctx = r.get("context") or {}
        size = (r.get("payload_summary") or {}).get("size_chars")
        ok, issues = audit_payload(ctx, case["student"], config, size_chars=size)
        if ok:
            passed += 1
        else:
            fails.append(f"[{case['id']}] " + "; ".join(issues))
    total = len(data["cases"])

    # cross-turn growth: a 6-turn S123 session must stay bounded
    _reset("S123", config)
    sess = orch.new_session()
    sizes = []
    for q in [
        "I am weak in Algebra, what should I study next?",
        "What should I study this week?",
        "Which topic should I prioritize first?",
        "How am I doing overall?",
        "I have a maths test coming up, help me prepare",
        "What should I focus on today?",
    ]:
        r = orch.respond(q, "S123", sess)
        sizes.append((r.get("payload_summary") or {}).get("size_chars", 0))
    grow_ok, grow_issues = audit_session_growth(sizes, config)
    total += 1
    if grow_ok:
        passed += 1
    else:
        fails.append("[session_growth] " + "; ".join(grow_issues) + f" sizes={sizes}")
    return passed, total, fails


def stage_adversarial(config: dict) -> tuple[int, int, list[str]]:
    data = _load("adversarial_queries.yaml")
    student = data.get("student", "S123")
    passed, fails = 0, []
    orch = _fresh(config)
    for case in data["cases"]:
        _reset(student, config)
        if case.get("inject"):
            _inject(orch, config)
        r = orch.respond(_query_of(case), student, orch.new_session())
        problems = []
        if "expect_intent" in case and r["intent"] not in case["expect_intent"]:
            problems.append(f"intent {r['intent']} not in {case['expect_intent']}")
        if case.get("forbid_tools") and r["tools_called"]:
            problems.append(f"tools were driven: {r['tools_called']}")
        if case.get("expect_grounded") and not r["grounding_ok"]:
            problems.append("expected grounded answer")
        for frag in case.get("forbid_text", []):
            if frag.lower() in r["answer"].lower():
                problems.append(f"leaked forbidden text '{frag}'")
        if problems:
            fails.append(f"[{case['id']}/{case['category']}] " + "; ".join(problems))
        else:
            passed += 1
    return passed, len(data["cases"]), fails


def stage_self_improvement(config: dict) -> tuple[int, int, list[str]]:
    pairs = _load("golden_dataset.yaml").get("self_improvement", [])
    passed, fails = 0, []
    orch = _fresh(config)
    for case in pairs:
        sid, focus = case["student"], case["focus_topic"]
        _reset(sid, config)
        sess = orch.new_session()
        orch.respond("what should I prioritize first?", sid, sess)
        before = _priority(sid, focus, config)
        orch.respond(f"the {focus} material really helped", sid, sess)
        after = _priority(sid, focus, config)
        if before is None or after is None:
            fails.append(f"[{case['id']}] could not read priority for {focus}")
        elif not (after < before):
            fails.append(f"[{case['id']}] {focus} priority did not drop: {before} -> {after}")
        else:
            passed += 1
    return passed, len(pairs), fails


def stage_persona(config: dict) -> tuple[int, int, list[str]]:
    cases = _load("golden_dataset.yaml").get("persona", [])
    passed, fails = 0, []
    orch = _fresh(config)
    for case in cases:
        sid, topic = case["student"], case["topic"]
        _reset(sid, config)
        # one positive feedback -> reflection must NOT set a preferred type yet (drift guard)
        s1 = orch.new_session()
        orch.respond(f"the {topic} material helped", sid, s1)
        orch.finalize_session(s1, sid)
        p1 = _persona(sid, config).get("preferred_material_type")
        # a second corroborating positive -> preferred type is now set
        s2 = orch.new_session()
        orch.respond(f"the {topic} material helped again", sid, s2)
        orch.finalize_session(s2, sid)
        p2 = _persona(sid, config).get("preferred_material_type")
        if p1 is not None:
            fails.append(f"[{case['id']}] persona set after a single event ({p1})")
        elif not p2:
            fails.append(f"[{case['id']}] persona not set after two corroborating events")
        else:
            passed += 1
    return passed, len(cases), fails


# --------------------------------------------------------------------------- #
# Helpers that read persisted memory                                           #
# --------------------------------------------------------------------------- #
def _priority(student_id: str, topic: str, config: dict) -> float | None:
    from src.memory.store import MemoryStore

    mastery = MemoryStore(student_id, config).load_mastery() or {}
    for key, rec in mastery.items():
        if key.lower() == topic.lower():
            return rec.get("priority_score")
    return None


def _persona(student_id: str, config: dict) -> dict:
    from src.memory.store import MemoryStore

    return MemoryStore(student_id, config).load_persona() or {}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    config = load_config()
    # This scorecard is the reproducibility proof for the deterministic approach: pin both
    # the mode and the backend so results are independent of a local Ollama. The LLM-centric
    # modes are evaluated separately by the NL suite (eval/phase1_runner.py), whose outcome
    # grading is semantic rather than exact-match.
    config.setdefault("pipeline", {})["mode"] = "deterministic"
    config.setdefault("llm", {})["backend"] = "deterministic"
    # ensure the index matches the full dataset before evaluating
    from src.utils.data_loader import load_dataset

    build_index(load_dataset("all").materials(), config)

    stages = [
        ("tool-call accuracy", stage_tool_calls),
        ("groundedness", stage_groundedness),
        ("context audit", stage_context_audit),
        ("adversarial robustness", stage_adversarial),
        ("self-improvement", stage_self_improvement),
        ("persona drift guard", stage_persona),
    ]

    print("=" * 68)
    print("AUREN EVALUATION SCORECARD")
    print("=" * 68)
    all_ok = True
    rows = []
    for name, fn in stages:
        passed, total, fails = fn(config)
        pct = 100.0 * passed / total if total else 100.0
        rows.append((name, passed, total, pct))
        status = "PASS" if passed == total else "FAIL"
        print(f"\n{name:24s}  {passed}/{total}  ({pct:5.1f}%)  [{status}]")
        for f in fails:
            all_ok = False
            print(f"    - {f}")

    print("\n" + "-" * 68)
    for name, _passed, _total, pct in rows:
        bar = "#" * int(pct / 5)
        print(f"  {name:24s} {pct:5.1f}%  {bar}")
    print("-" * 68)

    if all_ok:
        print("\nAll evaluation dimensions passed at 100%.")
        return 0
    print("\nEVALUATION FAILED — see failures above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
