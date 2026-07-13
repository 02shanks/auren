"""Tool: get_weak_topics — the student's weak topics with subject-score context."""

from src.tools.base import ToolContext
from src.utils.data_loader import DataIntegrityError

SPEC = {
    "type": "function",
    "function": {
        "name": "get_weak_topics",
        "description": (
            "Return the active student's weak topics, cross-referenced with their subject "
            "scores. Use when the student mentions struggling/weakness or asks what to improve. "
            "Returns an empty list (never invents a weakness) if none are on record."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "The active student's id."}
            },
            "required": ["student_id"],
        },
    },
}


def run(ctx: ToolContext, **kwargs) -> dict:
    student_id = kwargs.get("student_id")
    if not student_id:
        return {"error": "missing_argument", "argument": "student_id"}
    try:
        rec = ctx.repo.get_student(student_id)
    except DataIntegrityError as exc:
        return {"error": "data_integrity_error", "student_id": student_id, "detail": str(exc)}
    if rec is None:
        return {"error": "student_not_found", "student_id": student_id}
    subj_of: dict[str, str] = {}
    for test in rec.tests:
        for t in test.topics:
            subj_of.setdefault(t, test.subject)
    scores = {p.subject: p.overall_score_percentage for p in rec.performance}
    weak = [
        {"topic": t, "subject": subj_of.get(t), "subject_score": scores.get(subj_of.get(t))}
        for t in rec.profile.weak_topics
    ]
    note = "no weak topics on record" if not weak else f"{len(weak)} weak topic(s)"
    return {"student_id": student_id, "weak_topics": weak, "count": len(weak), "note": note}
