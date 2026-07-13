"""Pipeline coordinator

One turn in ``pipeline.mode`` deterministic/hybrid, end to end:
  route -> validate -> execute base tools (session-cached) -> recompute mastery
  -> compose recommend_study_material from the ranking/intent -> assemble context
  -> generate -> grounding check (regen once, then deterministic fallback)
  -> persist mastery + log episode/context.

``pipeline.mode: llm`` instead delegates to ``_respond_llm``/``AgenticExecutor``
(src/llm/agentic.py) — a native multi-round tool-calling loop with no deterministic
pre-router at all; only tool execution, mastery, context assembly, grounding, and
persistence are shared with the flow above.

Kept out of ``cli.py`` so the CLI stays a thin shell. ``today`` can be
pinned for deterministic tests/eval.
"""

import datetime as dt
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.llm.base import ToolCall, get_llm_client
from src.llm.deterministic import synthesize
from src.llm.prompts import clean_answer
from src.memory.cache import get_cache
from src.memory.mastery import MasteryEngine
from src.memory.store import MemoryStore
from src.orchestrator.context_assembler import assemble_context
from src.orchestrator.grounding import check_grounding
from src.orchestrator.router import route
from src.orchestrator.validator import validate
from src.retrieval.indexer import get_retriever
from src.tools.base import ToolContext
from src.tools.registry import dispatch, tool_specs
from src.utils.config import load_config, resolve_mode
from src.utils.logging_config import get_logger

log = get_logger("pipeline")
trace = get_logger("auren.trace")  # per-turn key details -> per-session log file

_READONLY = {
    "get_weak_topics",
    "get_upcoming_tests",
    "get_performance_summary",
    "recommend_study_material",
}
_NEEDS_TOOLS = {
    "weakness_focus",
    "weakness_list",
    "study_plan",
    "prioritize",
    "test_prep",
    "performance_query",
    "feedback",
}


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[dict] = field(default_factory=list)
    mastery_cache: dict[str, dict] = field(default_factory=dict)
    recommended: list[tuple[str, str]] = field(default_factory=list)
    cache: Any = None


def _norm(s: str) -> str:
    import re

    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


class Orchestrator:
    def __init__(
        self, config: dict | None = None, dataset: str = "all", today: dt.date | None = None
    ):
        self.config = config or load_config()
        self.dataset = dataset
        self.today = today
        from src.utils.data_loader import load_dataset

        self.repo = load_dataset(dataset)
        self.mastery = MasteryEngine(self.config)
        self.retriever = get_retriever(self.config, dataset, materials=self.repo.materials())
        self.ctx = ToolContext(
            repo=self.repo,
            retriever=self.retriever,
            mastery=self.mastery,
            config=self.config,
            today=today,
        )
        self.client = get_llm_client(self.config, repo=self.repo)
        self.tools = tool_specs()
        self.mode = resolve_mode(self.config)
        self.agentic = None
        if self.mode == "llm":
            from src.llm.agentic import AgenticExecutor

            # fails fast (clear message) if the configured backend can't do raw chat
            self.agentic = AgenticExecutor(self.client, self.config, self._dispatch, self.tools)

    def new_session(self) -> Session:
        return Session(cache=get_cache(self.config))

    # ---- tool execution with per-session caching ----
    def _dispatch(self, call: ToolCall, session: Session) -> dict:
        if call.name in _READONLY and session.cache is not None:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
            cached = session.cache.get(key)
            if cached is not None:
                return cached
            out = dispatch(call.name, self.ctx, call.arguments)
            session.cache.set(key, out)
            return out
        return dispatch(call.name, self.ctx, call.arguments)

    def _mastery_for(self, student_id: str, session: Session, reload: bool = False) -> dict:
        if reload or student_id not in session.mastery_cache:
            store = MemoryStore(student_id, self.config)
            mastery = store.load_mastery()
            rec = self._safe_record(student_id)
            if rec is not None:
                mastery = self.mastery.recompute(rec, mastery, today=self.ctx.now())
                store.save_mastery(mastery)  # persist current scores as durable memory
            session.mastery_cache[student_id] = mastery
        return session.mastery_cache[student_id]

    def _safe_record(self, student_id: str):
        from src.utils.data_loader import DataIntegrityError

        try:
            return self.repo.get_student(student_id)
        except DataIntegrityError:
            return None

    # ---- composition: recommend calls derived from ranking/intent ----
    def _compose(
        self, selection, outputs: list[dict], mastery: dict, student_id: str
    ) -> list[dict]:
        topics: list[str] = []
        intent = selection.intent
        if intent == "weakness_focus" and selection.focus_topic:
            topics.append(selection.focus_topic)
        elif intent in ("study_plan", "prioritize"):
            ranked = MasteryEngine.ranked_topics(mastery)
            n = 1 if intent == "prioritize" else 2
            topics.extend(t for t, _ in ranked[:n])
        elif intent == "test_prep":
            tests = next((o["output"] for o in outputs if o["tool"] == "get_upcoming_tests"), {})
            wt = next((o["output"] for o in outputs if o["tool"] == "get_weak_topics"), {})
            weak = {_norm(w["topic"]) for w in wt.get("weak_topics", [])}
            for test in sorted(
                tests.get("upcoming_tests", []), key=lambda t: t.get("days_until", 0)
            ):
                for tp in test.get("topics", []):
                    if _norm(tp) in weak:
                        topics.append(tp)
        recs: list[dict] = []
        seen: set[str] = set()
        for topic in topics:
            if _norm(topic) in seen:
                continue
            seen.add(_norm(topic))
            out = dispatch("recommend_study_material", self.ctx, {"topic": topic, "top_k": 1})
            recs.append({"tool": "recommend_study_material", "output": out})
        return recs

    def _mark_recommended_now(self, student_id: str, session: Session) -> None:
        """Mark topics recommended this session (affects next session's staleness)."""
        topics = sorted({t for sid, t in session.recommended if sid == student_id})
        rec = self._safe_record(student_id)
        if rec is None or not topics:
            return
        store = MemoryStore(student_id, self.config)
        mastery = store.load_mastery() or self.mastery.recompute(rec, {}, today=self.ctx.now())
        for t in topics:
            mastery = self.mastery.mark_recommended(mastery, t, today=self.ctx.now())
        mastery = self.mastery.recompute(rec, mastery, today=self.ctx.now())
        store.save_mastery(mastery)

    def finalize_session(self, session: Session, student_id: str) -> tuple[str, dict]:
        """End-of-session reflection + staleness bookkeeping (sec 7.3)."""
        from src.memory.reflection import run_reflection

        self._mark_recommended_now(student_id, session)
        store = MemoryStore(student_id, self.config)
        return run_reflection(store, session.session_id, self.config)

    # ---- the turn ----
    def respond(self, query: str, student_id: str, session: Session | None = None) -> dict:
        session = session or self.new_session()
        student_id = self.repo.canonical_id(
            student_id
        )  # tolerate whitespace/case, keep keys consistent
        if self.mode == "llm":
            return self._respond_llm(query, student_id, session)
        t0 = time.perf_counter()
        trace.info("query student=%s q=%r", student_id, (query or "")[:160])
        selection, notes = route(query, student_id, self.client, self.tools, self.repo, self.config)

        if selection.direct_answer is not None:
            trace.info(
                "direct-answer intent=%s (guardrail: no tools, no generation) total_ms=%.0f",
                selection.intent,
                (time.perf_counter() - t0) * 1000.0,
            )
            return self._finish(
                session,
                student_id,
                query,
                selection.intent,
                [],
                selection.direct_answer,
                notes,
                grounding_ok=True,
                fallback=False,
                payload_summary=None,
            )

        calls, vnotes = validate(selection, student_id, self.config)
        notes += vnotes

        # retry-then-deterministic-fallback if a model under-selected for a tool-needing intent
        if not calls and selection.intent in _NEEDS_TOOLS:
            from src.llm.deterministic import DeterministicClient

            det = DeterministicClient(self.config, self.repo).select_tools(
                query, student_id, self.tools
            )
            if det.direct_answer is not None:
                return self._finish(
                    session,
                    student_id,
                    query,
                    det.intent,
                    [],
                    det.direct_answer,
                    notes,
                    grounding_ok=True,
                    fallback=False,
                    payload_summary=None,
                )
            selection = det
            calls, vnotes = validate(selection, student_id, self.config)
            notes += ["fell back to deterministic selection", *vnotes]

        outputs = [{"tool": c.name, "output": self._dispatch(c, session)} for c in calls]

        did_feedback = any(c.name == "log_feedback" for c in calls)
        mastery = self._mastery_for(student_id, session, reload=did_feedback)

        recs = self._compose(selection, outputs, mastery, student_id)
        outputs += recs
        for o in recs:
            tp = o["output"].get("topic")
            if tp:
                session.recommended.append((student_id, tp))

        ranked = MasteryEngine.ranked_topics(mastery)
        persona = MemoryStore(student_id, self.config).load_persona()
        session_turns = session.turns + [
            {"student_id": student_id, "query": query, "intent": selection.intent}
        ]
        payload = assemble_context(
            query,
            selection.intent,
            outputs,
            ranked,
            session_turns,
            persona,
            self.config,
            focus_topic=selection.focus_topic,
        )

        ctx_dict = payload.to_llm_dict()
        t_gen = time.perf_counter()
        answer = self.client.generate(query, ctx_dict)
        ok, problems = check_grounding(answer, ctx_dict)
        fallback = False
        if not ok:
            retry = self.client.generate(query, ctx_dict)
            ok2, problems2 = check_grounding(retry, ctx_dict)
            if ok2:
                answer, ok = retry, True
            else:
                answer = synthesize(ctx_dict)
                ok, fallback = check_grounding(answer, ctx_dict)[0], True
                log.warning(
                    "grounding failed (%s); used deterministic fallback", problems2 or problems
                )
        gen_ms = (time.perf_counter() - t_gen) * 1000.0

        # student-facing cleanup: strip any internal meta-annotations a model added.
        # Done AFTER grounding so a hallucinated "(score: 99)" is still caught first.
        answer, stripped = clean_answer(answer)

        tools_called = [c.name for c in calls] + ["recommend_study_material"] * len(recs)
        self._trace_turn(
            student_id,
            selection.intent,
            tools_called,
            recs,
            ranked,
            payload,
            gen_ms=gen_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            grounded=ok,
            fallback=fallback,
            stripped=stripped,
        )
        return self._finish(
            session,
            student_id,
            query,
            selection.intent,
            outputs,
            answer,
            notes,
            grounding_ok=ok,
            fallback=fallback,
            payload_summary=payload.log_summary(),
            tools_called=tools_called,
            context_dict=ctx_dict,
        )

    def _respond_llm(self, query: str, student_id: str, session: Session) -> dict:
        """End-to-end LLM turn (pipeline mode 'llm'): the model routes, calls tools
        natively across rounds, and writes the answer. Structural validation,
        grounding verification, and all memory writes are shared with the other modes."""
        t0 = time.perf_counter()
        trace.info("query student=%s mode=llm q=%r", student_id, (query or "")[:160])

        mastery = self._mastery_for(student_id, session)
        ranked = MasteryEngine.ranked_topics(mastery)
        persona = MemoryStore(student_id, self.config).load_persona()

        res = self.agentic.run_turn(
            query, student_id, session.turns, ranked, persona, self.ctx.now(), session
        )

        if any(c.name == "log_feedback" for c in res.tool_calls):
            mastery = self._mastery_for(student_id, session, reload=True)
            ranked = MasteryEngine.ranked_topics(mastery)
        for o in res.outputs:
            if o["tool"] == "recommend_study_material" and o["output"].get("topic"):
                session.recommended.append((student_id, o["output"]["topic"]))

        session_turns = session.turns + [
            {"student_id": student_id, "query": query, "intent": res.intent}
        ]
        payload = assemble_context(
            query, res.intent, res.outputs, ranked, session_turns, persona, self.config
        )
        ctx_dict = payload.to_llm_dict()
        self.agentic.ground(res, ctx_dict, messages_note=f"Student question: {query}")
        answer, stripped = clean_answer(res.answer)

        tools_called = [c.name for c in res.tool_calls]
        self._trace_turn(
            student_id,
            res.intent,
            tools_called,
            [o for o in res.outputs if o["tool"] == "recommend_study_material"],
            ranked,
            payload,
            gen_ms=(time.perf_counter() - t0) * 1000.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            grounded=res.grounding_ok,
            fallback=res.grounded_fallback,
            stripped=stripped,
        )
        return self._finish(
            session,
            student_id,
            query,
            res.intent,
            res.outputs,
            answer,
            res.notes + [f"agentic rounds={res.rounds}"],
            grounding_ok=res.grounding_ok,
            fallback=res.grounded_fallback,
            payload_summary=payload.log_summary(),
            tools_called=tools_called,
            context_dict=ctx_dict,
        )

    def _trace_turn(
        self,
        student_id,
        intent,
        tools_called,
        recs,
        ranked,
        payload,
        *,
        gen_ms,
        total_ms,
        grounded,
        fallback,
        stripped,
    ) -> None:
        """Log the key per-turn details (tools, retrieval scores, memory, context, timing)
        to the session log. Deliberately compact — a few lines, not a full dump."""
        retrieval = []
        for o in recs:
            out = o["output"]
            top = (out.get("recommendations") or [{}])[0]
            score = top.get("score")
            retrieval.append(
                {
                    "topic": out.get("topic"),
                    "material_id": top.get("material_id"),
                    "match_type": top.get("match_type"),
                    "score": round(score, 3) if isinstance(score, (int, float)) else None,
                }
            )
        trace.info(
            "turn student=%s intent=%s backend=%s tools=%s gen_ms=%.0f total_ms=%.0f "
            "grounded=%s fallback=%s",
            student_id,
            intent,
            self.client.name,
            tools_called,
            gen_ms,
            total_ms,
            grounded,
            fallback,
        )
        if retrieval:
            trace.info("  retrieval=%s", json.dumps(retrieval, ensure_ascii=False))
        trace.info("  mastery_top=%s", [[t, round(s, 3)] for t, s in ranked[:5]])
        trace.info("  context=%s", json.dumps(payload.log_summary(), ensure_ascii=False))
        if stripped:
            trace.info("  stripped_meta=%s", stripped)

    def _finish(
        self,
        session,
        student_id,
        query,
        intent,
        outputs,
        answer,
        notes,
        *,
        grounding_ok,
        fallback,
        payload_summary,
        tools_called=None,
        context_dict=None,
    ) -> dict:
        store = MemoryStore(student_id, self.config)
        # topics touched + feedback details for the episodic log / persona inference
        topics: list[str] = []
        signal = material_type = None
        for o in outputs:
            out = o["output"]
            if o["tool"] == "log_feedback" and "updated_priority_score" in out:
                topics.append(out.get("topic"))
                signal, material_type = out.get("signal"), out.get("material_type")
            if o["tool"] == "recommend_study_material" and out.get("topic"):
                topics.append(out["topic"])
        episode = {
            "session_id": session.session_id,
            "intent": intent,
            "query": query[:500],
            "topics": [t for t in topics if t],
            "tools": tools_called or [o["tool"] for o in outputs],
            "response_summary": (answer or "")[:280],
        }
        if signal:
            episode["signal"] = signal
        if material_type:
            episode["material_type"] = material_type
        store.append_episode(episode)
        if payload_summary is not None:
            store.append_context_log({"session_id": session.session_id, **payload_summary})
        session.turns.append(
            {
                "student_id": student_id,
                "query": query,
                "intent": intent,
                # bounded answer echo so LLM-mode turns can carry conversation history
                "answer": (answer or "")[:400],
            }
        )
        return {
            "answer": answer,
            "intent": intent,
            "tools_called": tools_called or [o["tool"] for o in outputs],
            "notes": notes,
            "grounding_ok": grounding_ok,
            "grounded_fallback": fallback,
            "payload_summary": payload_summary,
            "context": context_dict,
        }
