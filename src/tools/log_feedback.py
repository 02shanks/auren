"""Tool: log_feedback — deterministic mastery-score update from a feedback signal."""

from src.memory.store import MemoryStore
from src.tools.base import ToolContext
from src.utils.data_loader import DataIntegrityError

_ALLOWED = {"helped", "not_helped", "positive", "negative", "up", "down", "good", "bad"}

SPEC = {
    "type": "function",
    "function": {
        "name": "log_feedback",
        "description": (
            "Record the student's feedback on a topic ('helped' or 'not_helped') and update that "
            "topic's mastery-priority score deterministically. Optionally pass the material_id the "
            "feedback refers to so the system can learn a material-type preference over time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "The active student's id."},
                "topic": {"type": "string", "description": "The topic the feedback is about."},
                "signal": {
                    "type": "string",
                    "description": "Feedback signal, e.g. 'helped' or 'not_helped'.",
                },
                "material_id": {
                    "type": "string",
                    "description": "Optional id of the material the feedback refers to.",
                },
            },
            "required": ["student_id", "topic", "signal"],
        },
    },
}


def run(ctx: ToolContext, **kwargs) -> dict:
    student_id = kwargs.get("student_id")
    topic = kwargs.get("topic")
    signal = kwargs.get("signal")
    material_id = kwargs.get("material_id")
    if not student_id:
        return {"error": "missing_argument", "argument": "student_id"}
    if not topic:
        return {"error": "missing_argument", "argument": "topic"}
    if signal not in _ALLOWED:
        return {"error": "invalid_signal", "allowed": ["helped", "not_helped"], "got": signal}
    try:
        rec = ctx.repo.get_student(student_id)
    except DataIntegrityError as exc:
        return {"error": "data_integrity_error", "student_id": student_id, "detail": str(exc)}
    if rec is None:
        return {"error": "student_not_found", "student_id": student_id}
    today = ctx.now()
    store = MemoryStore(student_id, ctx.config)
    mastery = store.load_mastery() or ctx.mastery.recompute(rec, {}, today=today)
    previous = mastery.get(topic, {}).get("priority_score")
    mastery = ctx.mastery.apply_feedback(mastery, topic, signal, today=today)
    mastery = ctx.mastery.recompute(rec, mastery, today=today)
    store.save_mastery(mastery)
    material_type = None
    if not material_id and getattr(ctx, "retriever", None) is not None:
        # infer the material the feedback most likely refers to (its canonical match for the topic)
        hits = ctx.retriever.recommend(topic, top_k=1)
        if hits:
            material_id = hits[0].get("material_id")
    if material_id:
        for m in ctx.repo.materials():
            if m.material_id == material_id:
                material_type = m.material_type
                break
    return {
        "student_id": student_id,
        "topic": topic,
        "signal": signal,
        "material_id": material_id,
        "material_type": material_type,
        "previous_priority_score": previous,
        "updated_priority_score": mastery.get(topic, {}).get("priority_score"),
        "ack": f"Recorded '{signal}' for {topic}.",
    }
