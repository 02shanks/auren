"""Deterministic offline engine — one of three LLMClient backends (the zero-setup,
model-free option; see src/llm/base.py), and ``pipeline.mode: deterministic``'s router.

This is a genuine rule-based system, not a lookup table keyed to the eval:

- ``select_tools`` classifies intent with generalizable keyword/pattern rules.
  Guardrails (instruction-override, system-prompt extraction, role-play,
  cross-student scope, academic integrity, off-topic, malformed) are checked FIRST
  and answer directly without calling tools. Business intents map to a base tool set;
  the orchestrator composes follow-up ``recommend_study_material`` calls.
- ``generate`` synthesizes the final answer purely from this turn's tool outputs plus
  the bounded memory slice, so every claim is grounded by construction. It doubles as
  the deterministic template fallback the grounding check falls back to (all three modes).

Because it is fully deterministic it also makes the eval reproducible and runnable with no
model present, while the same route -> validate -> generate orchestration (deterministic and
hybrid modes only) runs a real model when ``llm.backend`` is set to ``ollama``/``openrouter``.
``pipeline.mode: llm`` uses a different, agentic orchestration instead (src/llm/agentic.py)
with no deterministic pre-router in the loop at all.
"""

import re
import string
from typing import Any

from src.llm import prompts
from src.llm.base import Selection, ToolCall
from src.utils.data_loader import DataIntegrityError

_ID_RE = re.compile(r"\b(S\d{2,}|SYN-\d+)\b", re.IGNORECASE)
_ASCII_PUNCT = str.maketrans({c: " " for c in string.punctuation})
_MATERIAL_RE = re.compile(r"\bM\d{2,}\b", re.IGNORECASE)

POSITIVE_FB = (
    "helped",
    "helpful",
    "useful",
    "worked",
    "clearer",
    "makes sense",
    "made sense",
    "got it now",
    "was great",
    "was good",
    "was perfect",
    "was excellent",
    "was clear",
    "really helped",
    "helped a lot",
    "helped me",
    "loved it",
    "love it",
    "cleared it up",
)
NEGATIVE_FB = (
    "didn't help",
    "did not help",
    "didnt help",
    "not helpful",
    "wasn't helpful",
    "wasnt helpful",
    "no help",
    "useless",
    "was useless",
    "confusing",
    "was confusing",
    "still lost",
    "didn't work",
    "did not work",
    "waste of time",
    "not useful",
)

_OFFTOPIC = (
    "weather",
    "capital of",
    "joke",
    "recipe",
    "cook",
    "football",
    "cricket",
    "stock price",
    "movie",
    "song lyrics",
    "who won",
    "translate this",
)
_ONTOPIC = (
    "study",
    "learn",
    "topic",
    "subject",
    "test",
    "exam",
    "quiz",
    "score",
    "prepare",
    "revise",
    "practice",
    "weak",
    "strong",
    "improve",
    "material",
    "prioriti",
    "focus",
    "homework",
    "grade",
    "chapter",
    # Hindi/Hinglish (graceful degradation for Devanagari queries)
    "कमजोर",
    "पढ़",
    "अध्ययन",
    "परीक्षा",
    "टेस्ट",
    "विषय",
    "तैयार",
)

_SUBJECT_ALIASES = {
    "math": "mathematics",
    "maths": "mathematics",
    "sci": "science",
    "sst": "social science",
    "socialscience": "social science",
    "eng": "english",
}


def _norm(text: str) -> str:
    return " ".join((text or "").translate(_ASCII_PUNCT).lower().split())


class DeterministicClient:
    def __init__(self, config: dict, repo: Any = None) -> None:
        self.config = config
        self.repo = repo
        self.name = "deterministic"

    # ---- known-data helpers (topic/subject extraction is grounded in the student) ----
    def _record(self, student_id: str):
        if self.repo is None:
            return None
        try:
            return self.repo.get_student(student_id)
        except DataIntegrityError:
            return None

    def _known_topics(self, student_id: str) -> list[str]:
        rec = self._record(student_id)
        topics: list[str] = []
        if rec is not None:
            topics += rec.profile.weak_topics + rec.profile.strong_topics
            for t in rec.tests:
                topics += t.topics
        if self.repo is not None:
            topics += [m.topic for m in self.repo.materials()]
        # de-dupe preserving order
        seen: set[str] = set()
        out: list[str] = []
        for t in topics:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _known_subjects(self, student_id: str) -> list[str]:
        rec = self._record(student_id)
        subs: list[str] = []
        if rec is not None:
            subs += [p.subject for p in rec.performance]
            subs += [t.subject for t in rec.tests]
        seen: set[str] = set()
        out: list[str] = []
        for s in subs:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _extract_topic(self, query: str, student_id: str) -> str | None:
        nq = _norm(query)
        best: str | None = None
        for t in self._known_topics(student_id):
            nt = _norm(t)
            if nt and nt in nq and (best is None or len(nt) > len(_norm(best))):
                best = t
        if best:
            return best
        m = re.search(r"\b(?:in|with|at|on|about|for)\s+([a-z0-9][a-z0-9 \-]{2,40})", query, re.I)
        if m:
            return m.group(1).strip(" .!?")
        return None

    def _extract_subject(self, query: str, student_id: str) -> str | None:
        nq = _norm(query)
        for s in self._known_subjects(student_id):
            head = _norm(s).split()[0] if _norm(s) else ""
            if head and head in nq:
                return s
        for alias, canon in _SUBJECT_ALIASES.items():
            if alias in nq:
                for s in self._known_subjects(student_id):
                    if canon in _norm(s):
                        return s
                return canon.title()
        return None

    # ---- guardrail detection (checked before any business intent) ----
    def _guardrail(self, query: str, student_id: str) -> Selection | None:
        raw = query or ""
        nq = _norm(raw)
        # malformed / oversized
        distinct = set(re.sub(r"\s", "", raw.lower()))
        if len(raw) > 2000 or (len(raw) > 60 and len(distinct) <= 2):
            return Selection(intent="malformed_input", direct_answer=prompts.DEGRADE_MALFORMED)
        # instruction override / system-prompt extraction / role-play jailbreak
        jb = (
            "ignore all previous",
            "ignore previous instruction",
            "disregard your instruction",
            "disregard all previous",
            "forget your instruction",
            "forget all previous",
            "you are now",
            "new instructions",
            "repeat everything above",
            "everything above this line",
            "repeat the text above",
            "system prompt",
            "your instructions",
            "your system prompt",
            "reveal your",
            "print your prompt",
            "pretend you",
            "act as",
            "roleplay",
            "role-play",
            "no restrictions",
            "developer mode",
            "jailbreak",
            "dan mode",
        )
        if any(p in nq for p in jb):
            return Selection(intent="guardrail_refuse", direct_answer=prompts.REFUSAL_JAILBREAK)
        # cross-student scope
        other_ids = [m.group(0).upper() for m in _ID_RE.finditer(raw)]
        if any(i != (student_id or "").upper() for i in other_ids):
            return Selection(intent="guardrail_scope", direct_answer=prompts.REFUSAL_SCOPE)
        if any(
            p in nq
            for p in (
                "all students",
                "every student",
                "list students",
                "other student",
                "another student",
                "everyone in your data",
                "all the students",
            )
        ):
            return Selection(intent="guardrail_scope", direct_answer=prompts.REFUSAL_SCOPE)
        # academic integrity — wanting the answers/solutions for a graded task, or explicit cheating
        toks = set(nq.split())
        answers_noun = bool(toks & {"answer", "answers", "solution", "solutions"})
        graded = bool(
            toks
            & {
                "test",
                "tests",
                "exam",
                "exams",
                "quiz",
                "quizzes",
                "homework",
                "essay",
                "essays",
                "assignment",
                "assignments",
                "paper",
            }
        )
        acquire = any(
            p in nq
            for p in (
                "give me",
                "just give",
                "gimme",
                "tell me the",
                "show me the",
                "send me",
                "provide the",
                "hand me",
                "hand over",
                "i need the",
                "answers to",
                "answers for",
                "answer to",
                "answer for",
                "what are the answer",
                "whats the answer",
                "what's the answer",
            )
        )
        cheat = any(w in nq for w in ("copy", "cheat", "plagiar"))
        explicit = any(
            p in nq
            for p in (
                "do my homework",
                "write my essay",
                "write the essay for me",
                "answer key",
                "solutions to the",
                "exact answers",
            )
        )
        if explicit or (graded and cheat) or (acquire and answers_noun and graded):
            return Selection(intent="academic_integrity", direct_answer=prompts.REDIRECT_INTEGRITY)
        # explicit off-topic
        if any(p in nq for p in _OFFTOPIC):
            return Selection(intent="off_topic", direct_answer=prompts.DECLINE_OFFTOPIC)
        return None

    # ---- public: tool selection ----
    def select_tools(self, query: str, student_id: str, tools: list[dict]) -> Selection:
        guard = self._guardrail(query, student_id)
        if guard is not None:
            return guard
        nq = _norm(query)

        # feedback
        neg = any(p in nq for p in (_norm(x) for x in NEGATIVE_FB))
        pos = (not neg) and any(p in nq for p in (_norm(x) for x in POSITIVE_FB))
        # if a specific material is named with a bare sentiment adjective, treat as feedback too
        _mat_pre = _MATERIAL_RE.search(query)
        if _mat_pre and not neg and not pos:
            if any(
                w in nq
                for w in (
                    "great",
                    "good",
                    "excellent",
                    "perfect",
                    "amazing",
                    "awesome",
                    "brilliant",
                    "fantastic",
                    "loved",
                    "love",
                )
            ):
                pos = True
            elif any(
                w in nq
                for w in ("bad", "useless", "terrible", "awful", "confusing", "hated", "waste")
            ):
                neg = True
        if neg or pos:
            topic = self._extract_topic(query, student_id)
            mat = _MATERIAL_RE.search(query)
            material_id = mat.group(0).upper() if mat else None
            if not topic and material_id and self.repo is not None:
                for m in self.repo.materials():
                    if m.material_id == material_id:
                        topic = m.topic
                        break
            if not topic:
                return Selection(
                    intent="feedback",
                    direct_answer="Happy to record that — which topic did it relate to?",
                )
            signal = "not_helped" if neg else "helped"
            args = {"student_id": student_id, "topic": topic, "signal": signal}
            if material_id:
                args["material_id"] = material_id
            return Selection(
                tool_calls=[ToolCall("log_feedback", args)], intent="feedback", focus_topic=topic
            )

        # test prep
        if any(w in nq for w in ("test", "exam", "परीक्षा", "टेस्ट")) and any(
            w in nq
            for w in (
                "coming",
                "upcoming",
                "prepare",
                "prep",
                "study for",
                "before",
                "help",
                "आ रह",
                "तैयार",
            )
        ):
            subject = self._extract_subject(query, student_id)
            args: dict[str, Any] = {"student_id": student_id}
            if subject:
                args["subject"] = subject
            return Selection(
                tool_calls=[
                    ToolCall("get_upcoming_tests", args),
                    ToolCall("get_weak_topics", {"student_id": student_id}),
                ],
                intent="test_prep",
                focus_subject=subject,
            )

        # explicit prioritization
        if any(w in nq for w in ("prioriti", "most important", "top priority")) or (
            "first" in nq and "study" in nq
        ):
            return Selection(tool_calls=self._plan_tools(student_id), intent="prioritize")

        # weakness focus: a named (known) topic + weakness/help framing -> focus that one topic
        weakness_word = any(
            w in nq
            for w in (
                "weak",
                "struggl",
                "bad at",
                "hard for me",
                "improve",
                "help me with",
                "कमजोर",
            )
        )
        known_norm = {_norm(t) for t in self._known_topics(student_id)}
        if weakness_word:
            topic = self._extract_topic(query, student_id)
            if topic and _norm(topic) in known_norm:
                return Selection(
                    tool_calls=[
                        ToolCall("get_weak_topics", {"student_id": student_id}),
                        ToolCall("get_upcoming_tests", {"student_id": student_id}),
                    ],
                    intent="weakness_focus",
                    focus_topic=topic,
                )
            # weakness framing without a known topic
            elif topic and _norm(topic) not in known_norm:
                return Selection(
                    tool_calls=[ToolCall("get_weak_topics", {"student_id": student_id})],
                    intent="weakness_focus",
                    focus_topic=topic,
                )

            return Selection(
                tool_calls=[ToolCall("get_weak_topics", {"student_id": student_id})],
                intent="weakness_list",
            )

        # general study plan
        if any(
            p in nq
            for p in (
                "study this week",
                "what should i study",
                "study next",
                "what to study",
                "study plan",
                "what do i study",
                "help me study",
            )
        ):
            return Selection(tool_calls=self._plan_tools(student_id), intent="study_plan")

        # performance
        if any(
            p in nq
            for p in (
                "how am i doing",
                "my score",
                "my performance",
                "my grades",
                "how are my",
                "performance summary",
            )
        ):
            subject = self._extract_subject(query, student_id)
            args = {"student_id": student_id}
            if subject:
                args["subject"] = subject
            return Selection(
                tool_calls=[ToolCall("get_performance_summary", args)], intent="performance_query"
            )

        # plain material request
        if any(w in nq for w in ("material", "resource", "recommend", "notes", "video")):
            topic = self._extract_topic(query, student_id)
            if topic:
                return Selection(intent="weakness_focus", focus_topic=topic)

        # fallback: on-topic -> study plan; otherwise decline
        if any(w in nq for w in _ONTOPIC):
            return Selection(tool_calls=self._plan_tools(student_id), intent="study_plan")
        return Selection(intent="off_topic", direct_answer=prompts.DECLINE_OFFTOPIC)

    @staticmethod
    def _plan_tools(student_id: str) -> list[ToolCall]:
        return [
            ToolCall("get_weak_topics", {"student_id": student_id}),
            ToolCall("get_upcoming_tests", {"student_id": student_id}),
            ToolCall("get_performance_summary", {"student_id": student_id}),
        ]

    # ---- public: grounded synthesis ----
    def generate(self, query: str, context: dict) -> str:
        return synthesize(context)


# --------------------------------------------------------------------------- #
# Deterministic grounded synthesizer                                          #
# --------------------------------------------------------------------------- #
def _find(context: dict, tool: str) -> dict | None:
    for item in context.get("tool_outputs", []):
        if item["tool"] == tool:
            return item["output"]
    return None


def _find_all(context: dict, tool: str) -> list[dict]:
    return [i["output"] for i in context.get("tool_outputs", []) if i["tool"] == tool]


def _not_found_reply(student_id: str) -> str:
    return f"I couldn't find any study records for student {student_id}."


def _rec_for_topic(recs: list[dict], topic: str) -> dict | None:
    nt = _norm(topic)
    for out in recs:
        if _norm(out.get("topic", "")) == nt and out.get("recommendations"):
            return out["recommendations"][0]
    return None


def synthesize(context: dict) -> str:
    intent = context.get("intent", "unknown")
    sid = context.get("student_id", "")

    # any base tool reporting the student is missing -> honest not-found
    for item in context.get("tool_outputs", []):
        if item["output"].get("error") == "student_not_found":
            return _not_found_reply(sid)
        if item["output"].get("error") == "data_integrity_error":
            return (
                f"I can't proceed for {sid}: the records contain a data-integrity problem "
                "(a duplicated student id), so I won't guess which record is correct."
            )

    if intent in ("weakness_focus",):
        return _synth_weakness(context)
    if intent in ("weakness_list",):
        return _synth_weakness_list(context)
    if intent in ("study_plan", "prioritize"):
        return _synth_plan(context, commit=(intent == "prioritize"))
    if intent == "test_prep":
        return _synth_test_prep(context)
    if intent == "performance_query":
        return _synth_performance(context)
    if intent == "feedback":
        return _synth_feedback(context)
    return (
        "I can help with your weak topics, upcoming tests, performance, or study material. "
        "What would you like to work on?"
    )


def _synth_weakness(context: dict) -> str:
    wt = _find(context, "get_weak_topics") or {}
    recs = _find_all(context, "recommend_study_material")
    focus = context.get("focus_topic") or ""
    weak_names = [w["topic"] for w in wt.get("weak_topics", [])]
    is_weak = any(_norm(focus) == _norm(w) for w in weak_names)
    parts: list[str] = []
    if is_weak:
        parts.append(f"{focus} is on your weak-topics list, so it's worth focused practice.")
    elif focus:
        parts.append(f"Here's what I found for {focus}.")
    top = _rec_for_topic(recs, focus) if focus else None
    if top is None and recs and recs[0].get("recommendations"):
        top = recs[0]["recommendations"][0]
    if top:
        conf = "a direct match" if top.get("match_type") == "exact" else "the closest match I have"
        parts.append(f'Start with "{top["title"]}" ({top["material_id"]}) — {conf}.')
    else:
        parts.append("I don't have matching study material on record for it yet.")
    tests = _find(context, "get_upcoming_tests")
    if tests:
        for t in tests.get("upcoming_tests", []):
            if any(_norm(focus) == _norm(x) for x in t.get("topics", [])):
                parts.append(
                    f'Heads up: your {t["subject"]} test "{t["test_name"]}" covers it in '
                    f"{t['days_until']} days ({t['date']})."
                )
                break
    return " ".join(parts)


def _synth_weakness_list(context: dict) -> str:
    wt = _find(context, "get_weak_topics") or {}
    weak = wt.get("weak_topics", [])
    if not weak:
        return "Good news — you have no weak topics on record right now. Keep it up!"
    named = []
    for w in weak:
        if w.get("subject_score") is not None:
            named.append(f"{w['topic']} ({w['subject']} {w['subject_score']:.0f}%)")
        else:
            named.append(w["topic"])
    return "Your weak topics are: " + "; ".join(named) + "."


def _ranked(context: dict) -> list[tuple[str, float]]:
    return [(t, float(s)) for t, s in (context.get("memory", {}).get("ranked_topics") or [])]


def _synth_plan(context: dict, commit: bool) -> str:
    ranked = _ranked(context)
    wt = _find(context, "get_weak_topics") or {}
    tests = _find(context, "get_upcoming_tests") or {}
    recs = _find_all(context, "recommend_study_material")
    weak_names = {_norm(w["topic"]) for w in wt.get("weak_topics", [])}
    soon = {}
    for t in tests.get("upcoming_tests", []):
        for tp in t.get("topics", []):
            soon.setdefault(_norm(tp), t)
    if not ranked:
        return (
            "I don't have enough to build a ranking yet. Tell me a topic you're finding hard "
            "and I'll recommend material."
        )
    picks = ranked[:1] if commit else ranked[:3]
    lines: list[str] = []
    lead = "Focus on this first:" if commit else "Here's what to prioritize, most important first:"
    lines.append(lead)
    for i, (topic, score) in enumerate(picks, 1):
        reasons: list[str] = []
        if _norm(topic) in weak_names:
            reasons.append("a flagged weak area")
        if _norm(topic) in soon:
            t = soon[_norm(topic)]
            reasons.append(f"tested in {t['days_until']} days")
        reason = f" ({', '.join(reasons)})" if reasons else ""
        line = f"{i}. {topic} — priority {score:.2f}{reason}."
        rec = _rec_for_topic(recs, topic)
        if rec:
            line += f' Study "{rec["title"]}" ({rec["material_id"]}).'
        lines.append(line)
    if commit and picks:
        lines.append(f"If you only do one thing today, do {picks[0][0]}.")
    return "\n".join(lines)


def _synth_test_prep(context: dict) -> str:
    tests = _find(context, "get_upcoming_tests") or {}
    wt = _find(context, "get_weak_topics") or {}
    recs = _find_all(context, "recommend_study_material")
    upcoming = sorted(tests.get("upcoming_tests", []), key=lambda t: t.get("days_until", 0))
    weak_names = {_norm(w["topic"]) for w in wt.get("weak_topics", [])}
    subj = tests.get("subject_filter")
    if not upcoming:
        past = [f for f in tests.get("filtered_out", []) if f.get("reason") == "past"]
        label = f"{subj} " if subj else ""
        if past:
            return (
                f"You have no upcoming {label}tests on record — the one I found "
                f"({past[0]['test_id']}, dated {past[0].get('date')}) is in the past, so I set it "
                "aside. Want me to help with your weak topics instead?"
            )
        return (
            f"You don't have any upcoming {label}tests on record. Want me to help you work on "
            "your weak topics anyway?"
        )
    lines: list[str] = []
    for t in upcoming:
        tested = t.get("topics", [])
        weak_tested = [tp for tp in tested if _norm(tp) in weak_names]
        head = (
            f'Your {t["subject"]} test "{t["test_name"]}" is in {t["days_until"]} days '
            f"({t['date']}), covering {', '.join(tested) if tested else 'unspecified topics'}."
        )
        lines.append(head)
        if weak_tested:
            lines.append(f"You're weakest on: {', '.join(weak_tested)} — prioritize those.")
            for tp in weak_tested:
                rec = _rec_for_topic(recs, tp)
                if rec:
                    lines.append(f'  - {tp}: "{rec["title"]}" ({rec["material_id"]}).')
                else:
                    lines.append(f"  - {tp}: no matching material on record yet.")
        else:
            lines.append("None of your flagged weak topics are on this test — review broadly.")
    return "\n".join(lines)


def _synth_performance(context: dict) -> str:
    ps = _find(context, "get_performance_summary") or {}
    subs = ps.get("subjects", [])
    if not subs:
        return "I don't have any performance records for you yet."
    scored = [
        f"{s['subject']} {s['overall_score_percentage']:.0f}%"
        for s in subs
        if s.get("overall_score_percentage") is not None
    ]
    parts = [
        "Your current scores: " + ", ".join(scored) + "." if scored else "Scores are on record."
    ]
    if ps.get("weak_topics"):
        parts.append("Weak areas: " + ", ".join(ps["weak_topics"]) + ".")
    if ps.get("strong_topics"):
        parts.append("Strengths: " + ", ".join(ps["strong_topics"]) + ".")
    return " ".join(parts)


def _synth_feedback(context: dict) -> str:
    fb = _find(context, "log_feedback") or {}
    if fb.get("error") == "invalid_signal":
        return "I can record 'helped' or 'not_helped' — which was it?"
    if not fb or "updated_priority_score" not in fb:
        return "Tell me the topic and whether it helped, and I'll update your plan."
    positive = fb.get("signal") in ("helped", "positive", "up", "good")
    verb = "helped" if positive else "wasn't helpful"
    line = f"Got it — recorded that {fb['topic']} {verb}."
    new = fb.get("updated_priority_score")
    old = fb.get("previous_priority_score")
    if new is not None and old is not None:
        line += f" Its priority moved from {old:.2f} to {new:.2f}."
    elif new is not None:
        line += f" Its priority is now {new:.2f}."
    return line
