"""Post-generation grounding verification.

The evidence pool is exactly what the model was shown: this turn's tool outputs plus
the bounded memory slice (ranked scores, persona) — both attributable.
Every date, material/test id, and number in the answer must trace back to that pool;
any student id in the answer must be the active one. A failure triggers one regen and
then a deterministic grounded fallback in the pipeline.
"""

import json
import re
from typing import Any

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ID_RE = re.compile(r"\b(?:M|T)\d{2,}\b", re.IGNORECASE)
_SID_RE = re.compile(r"\b(?:S\d{2,}|SYN-\d+)\b", re.IGNORECASE)
_LIST_ORDINAL_RE = re.compile(r"(?m)^\s*\d+\.\s")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

_NUM_TOL = 0.01


def _evidence_text(context: dict[str, Any]) -> str:
    parts = [json.dumps(context.get("tool_outputs", []), ensure_ascii=False)]
    parts.append(json.dumps(context.get("memory", {}), ensure_ascii=False))
    parts.append(json.dumps(context.get("persona", {}), ensure_ascii=False))
    return " ".join(parts)


def _numbers(text: str) -> list[float]:
    text = _DATE_RE.sub(" ", text)
    text = _ID_RE.sub(" ", text)
    text = _SID_RE.sub(" ", text)
    text = _LIST_ORDINAL_RE.sub(" ", text)
    out: list[float] = []
    for tok in _NUM_RE.findall(text):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def check_grounding(answer: str, context: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    evidence = _evidence_text(context)
    active = str(context.get("student_id", "")).upper()

    # student ids: must be the active student (foreign id => cross-student leak)
    for sid in {m.group(0).upper() for m in _SID_RE.finditer(answer)}:
        if sid != active:
            problems.append(f"foreign student id in answer: {sid}")

    # dates
    ev_dates = set(_DATE_RE.findall(evidence))
    for d in set(_DATE_RE.findall(answer)):
        if d not in ev_dates:
            problems.append(f"ungrounded date: {d}")

    # material / test ids
    ev_ids = {i.upper() for i in _ID_RE.findall(evidence)}
    for i in {m.group(0).upper() for m in _ID_RE.finditer(answer)}:
        if i not in ev_ids:
            problems.append(f"ungrounded id: {i}")

    # numbers (tolerant float match)
    ev_nums = _numbers(evidence)
    for n in _numbers(answer):
        if not any(abs(n - e) <= _NUM_TOL for e in ev_nums):
            problems.append(f"ungrounded number: {n:g}")

    return (len(problems) == 0), problems
