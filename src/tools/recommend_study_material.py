"""Tool: recommend_study_material — exact-then-semantic material match (RAG as one tool).

Material title/topic fields are returned verbatim as *data*. This tool never
executes instruction-like text found inside a material field.
"""

from src.tools.base import ToolContext

SPEC = {
    "type": "function",
    "function": {
        "name": "recommend_study_material",
        "description": (
            "Find study material for a topic. Tries an exact/structured match first, then a "
            "semantic fallback; below a similarity threshold it honestly returns no match rather "
            "than inventing one. Each result carries match_type ('exact'|'semantic') and a score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic to find material for."},
                "top_k": {"type": "integer", "description": "Max results (default 3)."},
            },
            "required": ["topic"],
        },
    },
}


def run(ctx: ToolContext, **kwargs) -> dict:
    topic = kwargs.get("topic")
    if not topic:
        return {"error": "missing_argument", "argument": "topic"}
    top_k = kwargs.get("top_k")
    try:
        k = int(top_k) if top_k not in (None, "") else None
    except (TypeError, ValueError):
        k = None
    recs = ctx.retriever.recommend(str(topic), k)
    note = (
        "no matching material found"
        if not recs
        else f"{len(recs)} match(es) via {recs[0]['match_type']}"
    )
    return {"topic": topic, "recommendations": recs, "count": len(recs), "note": note}
