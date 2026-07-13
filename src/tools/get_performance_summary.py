"""Tool: get_performance_summary — subject scores + weak/strong topic context."""

from src.tools.base import ToolContext
from src.utils.data_loader import DataIntegrityError
from src.utils.subjects import subject_matches

SPEC = {
    "type": "function",
    "function": {
        "name": "get_performance_summary",
        "description": (
            "Return the student's per-subject scores (and trend if available), plus their "
            "weak and strong topics for context. Optionally filter by subject."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "The active student's id."},
                "subject": {"type": "string", "description": "Optional subject filter."},
            },
            "required": ["student_id"],
        },
    },
}


def run(ctx: ToolContext, **kwargs) -> dict:
    student_id = kwargs.get("student_id")
    subject = kwargs.get("subject")
    if not student_id:
        return {"error": "missing_argument", "argument": "student_id"}
    try:
        rec = ctx.repo.get_student(student_id)
    except DataIntegrityError as exc:
        return {"error": "data_integrity_error", "student_id": student_id, "detail": str(exc)}
    if rec is None:
        return {"error": "student_not_found", "student_id": student_id}
    subs = [
        {
            "subject": p.subject,
            "overall_score_percentage": p.overall_score_percentage,
            "trend": p.trend,
        }
        for p in rec.performance
        if subject_matches(subject, p.subject)
    ]
    note = "no performance records" if not subs else f"{len(subs)} subject(s)"
    return {
        "student_id": student_id,
        "subject_filter": subject,
        "subjects": subs,
        "weak_topics": rec.profile.weak_topics,
        "strong_topics": rec.profile.strong_topics,
        "count": len(subs),
        "note": note,
    }
