"""Tool: get_upcoming_tests — future tests only (past + unparseable dates filtered)."""

from src.tools.base import ToolContext
from src.utils.data_loader import DataIntegrityError
from src.utils.dates import parse_date
from src.utils.subjects import subject_matches

SPEC = {
    "type": "function",
    "function": {
        "name": "get_upcoming_tests",
        "description": (
            "Return the student's upcoming tests with days-until, sorted soonest first. "
            "Past-dated and unparseable-date tests are filtered out and reported separately. "
            "Optionally filter by subject."
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
    today = ctx.now()
    kept: list[dict] = []
    filtered: list[dict] = []
    for t in rec.tests:
        if not subject_matches(subject, t.subject):
            continue
        d = parse_date(t.date)
        if d is None:
            filtered.append(
                {"test_id": t.test_id, "reason": "unparseable_date", "raw_date": t.date}
            )
            continue
        du = (d - today).days
        if du < 0:
            filtered.append({"test_id": t.test_id, "reason": "past", "date": t.date})
            continue
        kept.append(
            {
                "test_id": t.test_id,
                "subject": t.subject,
                "test_name": t.test_name,
                "date": t.date,
                "days_until": du,
                "topics": t.topics,
            }
        )
    kept.sort(key=lambda x: x["days_until"])
    note = "no upcoming tests" if not kept else f"{len(kept)} upcoming test(s)"
    return {
        "student_id": student_id,
        "subject_filter": subject,
        "upcoming_tests": kept,
        "filtered_out": filtered,
        "count": len(kept),
        "note": note,
    }
