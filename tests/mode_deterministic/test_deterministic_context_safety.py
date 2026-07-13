"""End-to-end context-safety checks driven through a real orchestrator turn in
pipeline.mode='deterministic': no foreign student data ever reaches the assembled context
(sec 8.4 isolation), and an ungrounded generation is caught and replaced by the deterministic
fallback (sec 8.3). These need a resolved turn (not just the pure ``assemble_context``/
``check_grounding`` functions covered in tests/shared/test_context_assembly.py), so they live
here where a fast, offline ``orch`` is guaranteed by tests/mode_deterministic/conftest.py.
"""

import json
import re


def test_context_carries_no_foreign_student_data(orch, reset_memory):
    reset_memory("S123")
    r = orch.respond("what should I study this week?", "S123", orch.new_session())
    blob = json.dumps(r["context"]["tool_outputs"], default=str, ensure_ascii=False)
    foreign = {m.upper() for m in re.findall(r"S\d{2,}|SYN-\d+", blob)} - {"S123"}
    assert not foreign


def test_ungrounded_generation_falls_back(orch, reset_memory, monkeypatch):
    reset_memory("S123")
    # force the "model" to hallucinate an unknown material id and an unsupported number
    monkeypatch.setattr(
        orch.client, "generate", lambda q, ctx: "Study Topic Z (M999) — your score is 999%."
    )
    r = orch.respond("what should I study this week?", "S123", orch.new_session())
    assert r["grounded_fallback"] is True
    assert r["grounding_ok"] is True  # the deterministic fallback is grounded by construction
    assert "M999" not in r["answer"] and "999%" not in r["answer"]
