"""pipeline.mode == 'deterministic' — curated results from the Phase 1 natural-language
evaluation (PHASE1_REPORT.md §3), pinned down as regression tests so this mode's specific
strengths and weaknesses stay visible and don't silently drift. These complement — not
duplicate — the canonical/adversarial/routing coverage in
tests/mode_deterministic/test_deterministic_e2e.py, test_deterministic_adversarial.py, and
test_deterministic_routing.py.

Every case here is graded exactly as recorded in eval/phase1_results_deterministic.jsonl —
this is the actual output of the real deterministic engine (not a fake/replay, since this mode
needs no model and is fast enough to run for real every time).
"""


def test_b6_revision_keyword_stem_mismatch_declines_incorrectly(orch, reset_memory):
    """Known weakness: the on-topic keyword list contains 'revise' but not 'revision' — a
    simple stemming gap. 'Give me a revision schedule for the next 7 days' is a legitimate
    study_plan request but declines as off_topic because 'revision' does not contain 'revise'
    as a substring. See PHASE1_REPORT.md §3.1 (paraphrase category) for the broader pattern
    this single case exemplifies."""
    reset_memory("S123")
    r = orch.respond(
        "Give me a revision schedule for the next 7 days", "S123", orch.new_session()
    )
    # documents the CURRENT (undesirable) behavior; llm mode does not have this gap —
    # see tests/mode_llm/test_llm_improvements_over_other_modes.py for comparable cases
    assert r["intent"] == "off_topic"


def test_c5_practice_material_request_is_not_misfired_as_academic_integrity(orch, reset_memory):
    """Known strength: the academic-integrity guardrail is well-scoped. 'What are the answers
    to the practice questions in M101?' contains 'answers to' (an acquisition phrase) but no
    graded-work token (test/exam/quiz/homework/...) — 'questions' alone isn't graded work — so
    it correctly falls through to a normal study_plan answer instead of a false-positive
    integrity refusal. Contrast with test_c7 and PHASE1_REPORT.md §3.2 for cases where the
    *guardrail* keyword lists ARE too broad; this one shows the *integrity* rule specifically
    is calibrated correctly."""
    reset_memory("S123")
    r = orch.respond(
        "What are the answers to the practice questions in M101?", "S123", orch.new_session()
    )
    assert r["intent"] != "academic_integrity"
    assert r["grounding_ok"]


def test_c7_class_average_question_declines_incorrectly(orch, reset_memory):
    """Known weakness: 'How do I compare with the class average?' is a legitimate performance
    question about the student's OWN standing (not a request for another student's data — no
    guardrail_scope trigger fires, correctly) — but no performance-query keyword matches
    'compare'/'class average', so it falls all the way through to off_topic instead of
    honestly explaining that only the student's own scores are available."""
    reset_memory("S123")
    r = orch.respond("How do I compare with the class average?", "S123", orch.new_session())
    assert r["intent"] == "off_topic"


def test_d1_compound_weakness_request_falsely_claims_no_material(orch, reset_memory):
    """Known weakness (PHASE1_REPORT.md §3.4, multi_intent): 'Show my weak areas and recommend
    material for each' is a compound request over THREE weak topics (Algebra -> M101,
    Quadratic Equations -> M103, Light - Reflection and Refraction -> M105 all exist in
    data/study_materials.json). The single-intent router picks 'weakness_focus' — a
    single-topic intent — so composition attempts exactly one recommend_study_material call for
    whatever (mis-extracted) topic it derived from the compound sentence, that lookup finds
    nothing, and the answer falsely claims no material exists at all, even though material
    exists for every one of the three topics actually being asked about."""
    reset_memory("S123")
    r = orch.respond(
        "Show my weak areas and recommend material for each", "S123", orch.new_session()
    )
    assert r["tools_called"].count("recommend_study_material") == 1  # only ONE topic attempted
    assert "don't have matching study material" in r["answer"]  # false: M101/M103/M105 all exist


def test_f9_multiple_tests_this_week_surfaces_via_priority_not_a_test_list(orch, reset_memory):
    """Partial success (PHASE1_REPORT.md §3.5, data_edge/SYN-12): 'what tests do I have this
    week?' has no 'test'+prep-word combination, so it misroutes to study_plan rather than
    test_prep — but the mastery-priority reasons happen to surface BOTH of SYN-12's same-week
    tests ('tested in 3 days' / 'tested in 5 days') as part of the ranked topics, so the
    information the student needs is present even though it's not framed as an explicit test
    list the way test_prep's template would present it."""
    reset_memory("SYN-12")
    r = orch.respond("what tests do I have this week?", "SYN-12", orch.new_session())
    assert r["intent"] == "study_plan"  # misrouted away from test_prep
    assert "3 days" in r["answer"] and "5 days" in r["answer"]  # but both tests still surfaced
