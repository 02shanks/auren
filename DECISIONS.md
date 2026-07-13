# Architecture decisions

A short log of the decisions that shaped Auren, each with the context, the choice, and the
trade-off. Read alongside the [README](README.md).

---

## D-01 — A deterministic engine is the default "LLM"

**Context.** The blueprint requires local/offline operation *and* a proof, via evaluation,
that the system is correct — in any environment, including CI with no GPU, no network, and no
model runtime.

**Decision.** Ship a genuine deterministic engine — a rule-based intent classifier
(`src/llm/deterministic.py`) and a grounded template synthesizer — as the default backend, and
select it automatically when no model is reachable (`llm.backend: auto`). Real backends
(`OllamaClient`, `OpenRouterClient`) implement the same `LLMClient` protocol and are used when
available.

**Consequences.** The product runs and the full eval passes with zero setup. Every other layer
— tools, memory, context assembly, grounding, validation, safety — is backend-agnostic and
exercised identically whether an LLM is present or not. The deterministic engine is the
*floor*, not the ceiling: a stronger model yields more fluent prose, and grounding keeps it
honest. The trade-off is that free-form phrasing outside the classifier's patterns is handled
less flexibly than an LLM would; this is an accepted limitation, documented in the README.

> **Update (see D-27).** This "deterministic is the default" framing was measured, not just
> assumed, in a later evaluation: across 67 natural-language test cases the classifier declined
> or misrouted the majority of paraphrased/conversational queries. That evaluation is what
> motivated D-27's three-mode pivot — deterministic remains available and is still the correct
> choice for zero-dependency or compliance-sensitive contexts, but is no longer positioned as
> this repository's recommended default.

---

## D-02 — Hashing embedder as the default, Ollama embeddings optional

**Context.** Semantic retrieval needs embeddings, but pulling a heavy embedding model (or
calling a network service) contradicts the offline-first goal and slows CI.

**Decision.** Default to a dependency-free `HashingEmbedder` (blake2b over char-trigrams and
word tokens, 512 dims, cosine similarity). Use `nomic-embed-text` via Ollama (`OllamaEmbedder`)
when the `local` extra and runtime are available.

**Consequences.** Retrieval works offline and deterministically, and genuinely captures lexical
similarity — the reworded material "Mitosis Cell Division" is recalled for the weak topic
"Cell Division and Mitosis", while an unrelated query stays below the 0.60 threshold and
returns nothing. The trade-off is weaker paraphrase handling than a learned embedder; the
Ollama path exists for teams that want it.

---

## D-03 — Vector store: in-memory default, Chroma optional, graceful fallback

**Context.** A persistent vector DB is useful at scale but is another heavy dependency and
another failure mode.

**Decision.** Default to a small in-process vector store; use ChromaDB when the `vector` extra
is installed. `get_vector_store` falls back to the in-memory implementation if Chroma is
unavailable, logging the fallback rather than failing.

**Consequences.** The common path has no external dependency; scaling up is a config/extra
change, not a code change. The index is cheap to rebuild, so an ephemeral store is not a
practical limitation for the dataset sizes here.

---

## D-06 — Local-first, single-student by construction

**Context.** The assistant handles one student's data and must not leak across students.

**Decision.** All state is local files under `memory/<student_id>/`. Isolation is enforced in
two independent places: the context assembler only ever receives the active student's memory,
and the validator forces `student_id = active` on every tool call before dispatch. Grounding
additionally rejects any answer containing a foreign student id.

**Consequences.** Cross-student leakage is prevented structurally, not just by prompt
instruction — a defense that holds even if the model misbehaves. This is verified by the
context-audit eval dimension and by `tests/test_context_assembly.py`.

---

## D-08 — Redis is designed for but not used

**Context.** The blueprint mentions caching. A distributed cache (Redis) is the obvious
production choice but is an external service.

**Decision.** Provide a cache seam (`src/memory/cache.py`) with an in-process implementation
and leave a Redis-backed option as a documented extra (`cache`), unused by default.

**Consequences.** No runtime dependency for the default experience; the seam means adding Redis
later is localized. For a single-user CLI, the in-process cache is sufficient.

---

## D-10 — Local runtime dependencies install by default

**Context.** `ollama`, `chromadb`, and `redis` are large and slow to install; most of the time
the model/vector services may not be running. Earlier versions put `ollama` and `chromadb`
behind extras, but that made plain `uv sync` remove the packages while the default config
still tried to use them.

**Decision.** Install the local runtime client libraries (`ollama`, `chromadb`) with a plain
`uv sync`, while keeping runtime fallback behavior if Ollama is unreachable or ChromaDB cannot
initialize. The historical `local` and `vector` extras remain as no-op compatibility aliases.

**Consequences.** The documented setup path matches the default config, so rebuilding the
environment does not silently disable local embeddings or vector storage. Installs are heavier,
but failures are clearer and the app still has deterministic fallbacks.

---

## D-12 — `focus_topic` is threaded end-to-end

**Context.** Weakness-focus and test-prep answers must center on a specific topic, and the
context must make that topic unambiguous to the generator.

**Decision.** The classifier extracts a `focus_topic` into the `Selection`; the pipeline passes
it into `assemble_context`, and it is carried in the context payload (`to_llm_dict`) so the
generator and grounding both see it.

**Consequences.** Topic-centered intents produce topic-centered answers, and grounding can
verify the focus is supported. Without this thread, the generator could drift to a different
weak topic.

---

## D-13 — Unicode-safe normalization (Devanagari)

**Context.** The dataset deliberately includes Devanagari content, and some Hindi/Hinglish
keywords aid graceful degradation. A naive regex-based punctuation strip corrupts combining
marks.

**Decision.** Normalize with `str.translate` over an explicit ASCII-punctuation table rather
than a regex character class, preserving non-ASCII combining characters.

**Consequences.** Devanagari topics and queries are handled without mojibake, and matching
still works. This is why the Devanagari edge-case student behaves correctly.

---

## D-14 — "Recommended this session" is applied at session finalize, not per turn

**Context.** Staleness in the mastery formula depends on when a topic was last recommended.
Applying that mark mid-session would let a topic's score drift *within* a single session and
muddy the canonical traces.

**Decision.** Collect recommended topics during the session and apply the "recommended" mark
once, in `finalize_session`, so it affects the *next* session's staleness, not the current one.

**Consequences.** Within-session ranking is stable and canonical query traces are clean, while
staleness still evolves across sessions as intended.

---

## D-15 — Baseline mastery is persisted at session start

**Context.** Self-improvement must be observable: a "before" score, feedback, then a lower
"after" score. If scores were only written on feedback, there would be no durable baseline to
compare against, and a fresh session would have nothing on disk.

**Decision.** `_mastery_for` recomputes and saves the current scores to the store at session
start, so mastery is durable memory from the first turn. Feedback then updates and re-persists
it; `finalize_session` applies staleness bookkeeping.

**Consequences.** The self-improvement eval dimension can read a real before/after from disk,
and the CLI reflects current scores across runs. The write is idempotent given the same inputs,
so it does not perturb determinism.

---

## D-16 — Retrieved text is data, and injection defense is layered

**Context.** Study-material titles and topics are attacker-controllable in the injection
fixtures and could carry instructions ("ignore all previous instructions…").

**Decision.** Three independent layers: (1) the system prompt states that tool-output and
material-field text is data, never instructions; (2) the deterministic synthesizer only ever
slots fields into fixed sentence templates, so field contents cannot alter control flow; and
(3) grounding blocks any injected foreign id or unsupported number from surfacing as fact.
Poisoned fixtures are kept separate from the clean dataset.

**Consequences.** Injected instructions are inert regardless of backend, demonstrated by the
adversarial eval dimension and `tests/test_adversarial.py`. Poisoned materials remain
*retrievable* (we do not silently hide data); the defense is in how their contents are treated.

---

## D-17 — Subject-name canonicalization for tool filters

**Context.** `test_q4` failed only under the Ollama backend. The LLM extracted `subject="Maths"`
from "I have a Maths test coming up" and called `get_upcoming_tests(subject="Maths")`. The tool's
filter was a naive substring test (`subject.lower() in t.subject.lower()`); since "maths" is not a
substring of "Mathematics", T201 was skipped *before* the past-date branch, so `filtered_out` came
back empty and the answer said "no upcoming Maths tests". The deterministic classifier happened to
call the tool without a subject, so it never hit the bug — which is why the offline suite was green
while the user's Ollama run failed. So this was a real pipeline bug on the subject-filtered path,
exposed (not caused) by the LLM.

**Decision.** Added `src/utils/subjects.py` with `canonical_subject()` (alias map: maths/math →
mathematics, sci → science, etc.) and `subject_matches()` (canonical equality or a shared canonical
word). Both `get_upcoming_tests` and `get_performance_summary` now filter through `subject_matches`.
The fix is data-driven and general (any shorthand/synonym), not special-cased to T201.

**Consequences.** "Maths", "Mathematics", "maths", and no-filter all now surface T201 in
`filtered_out` as `past`, so both backends' context and answers are correct. `test_q4` was also
rewritten to assert the backend-independent structured invariant (the tool marks T201 `past`) plus a
flexible prose check, so it holds under the LLM *and* the deterministic engine rather than pinning
exact model wording.

---

## D-18 — Student-facing output: prompt hardening + a sanitizer safety net

**Context.** Under Ollama the answer contained developer-facing annotations — "(from memory and
tool data)", "(exact match)", "(strong topic)", "(weak topic)", "(score: 52.0)". These come from the
model, not the deterministic synthesizer (which already uses clean templates). A prompt instruction
alone is not reliable on a small local model.

**Decision.** Two layers. (1) The system/generation prompts now specify a student-facing format and
explicitly forbid source/match annotations and raw scores, with a good/bad example. (2)
`prompts.clean_answer()` strips the known meta-annotation patterns from any generated answer as a
backend-agnostic safety net. It is applied in the pipeline *after* grounding, so a hallucinated
"(score: 99)" is still caught by the grounding check before the annotation is removed. The
diagnostic detail the labels tried to convey (match type, retrieval score, priority) is written to
the per-session trace log instead (see D-19). Material ids like `(M101)` and percentages like `52%`
are preserved — they are useful to the student and grounded.

**Consequences.** The student sees clean prose regardless of how chatty the model is; internal
detail is still available for debugging in the logs. The deterministic engine is unchanged (its
output was already clean and the eval asserts its "priority" phrasing), so accuracy is unaffected.

---

## D-19 — Per-session trace logging for the CLI

**Context.** Debugging a turn (why this answer? which tools/materials/scores?) needs a compact,
per-run record — not a firehose, and not noise on the CLI's stdout, which must stay answer-only.

**Decision.** `logging_config` now sets the root logger to DEBUG and lets each handler filter
independently: the stderr handler stays at WARNING (console shows only warnings/errors), while
`start_session_log(session_id)` attaches an INFO `FileHandler` at `logs/<session_id>/session.log`.
The CLI opens the session log at start (printing its path to stderr) and closes it in a `finally`.
The pipeline emits one compact block per turn via an `auren.trace` logger: intent, backend, tools,
`gen_ms`/`total_ms`, grounded/fallback, retrieval `{topic, material_id, match_type, score}`, the
top mastery scores, the context summary (sizes, tools-in-context, persona flag, payload hash), and
any stripped meta-annotations.

**Consequences.** Every CLI session leaves a self-contained trace that ties the final answer back to
tools, memory, retrieval scores, backend, and timing, at INFO granularity, without polluting stdout
or the answer.

---

## D-20 — Deterministic backend pinned for tests and eval; Ollama is opt-in

**Context.** With `backend: auto`, running the suite on a machine with Ollama up sent every test
through the LLM — slow (the failing run took ~38 s for one test) and non-deterministic, which is how
`test_q4`'s brittle prose assertion surfaced.

**Decision.** `tests/conftest.py` pins `backend: deterministic` by default (overridable with
`AUREN_TEST_BACKEND=ollama`), and `eval/run_eval.py` forces `deterministic` because the eval is the
reproducible correctness proof (D-01). Assertions were written to be backend-agnostic (structured
invariants + flexible prose), so the same suite still passes under Ollama.

**Consequences.** CI-style runs are fast and reproducible regardless of a local Ollama; the LLM path
can still be exercised on demand with the env var, and is additionally covered by CLI smoke runs and
a mock-LLM check of the generate → ground → clean → log flow.

> **Update (see D-28).** This decision's described mechanism had regressed by the time of this
> engagement: `tests/conftest.py`'s `orch`/`config` fixtures called `load_config()` with no
> backend override at all, and no `AUREN_TEST_BACKEND` env var existed anywhere in the
> codebase — verified directly: a single "unit" test using the `orch` fixture measured **69.5
> seconds** with a local Ollama server reachable, proving it was silently driving live model
> calls. D-28 replaces the single blanket pin this decision described with an explicit
> per-mode-directory pin instead.

---

## D-21 — Ollama + ChromaDB primary, deterministic fallback (and a sandbox note)

**Context.** `chromadb` and `ollama` are now primary dependencies: the intended runtime is local
Ollama models (chat `qwen3:8b`, embeddings `nomic-embed-text`) with ChromaDB for persistent vectors,
degrading to the offline engine when a service is absent. The retrieval threshold was raised to 0.60.

**Decision.** Kept `backend: auto` / `embedder: auto` / `vector_store: auto` so a plain `uv sync`
installs the libraries and the app prefers Ollama+Chroma but falls back cleanly. The Issue A/B/C
fixes are all backend-agnostic so they apply equally to the Ollama path.

**Consequences.** `uv sync` installs chromadb 1.5.9 + the Ollama client, and **ChromaDB is the
active vector store** end-to-end (verified). The build sandbox's egress allowlist (PyPI/npm/GitHub/
Ubuntu only) blocks `registry.ollama.ai` (`host_not_allowed`), so Ollama model *weights* cannot be
pulled here; the Ollama code path is therefore verified by the clean fallback behavior, a mock-LLM
end-to-end run, and the backend-agnostic tests, and will use real models unchanged in an environment
where Ollama is reachable. At threshold 0.60 the reworded-topic semantic case (SYN-16) still matches
via the hashing embedder + Chroma; the offline suite and eval remain green.

---

## D-22 — Academic-integrity detection is token-based, not phrase-brittle (QA it.1)

**Context.** QA found "just give me the exam answers so I can copy them" routed to `study_plan`
and ran tools: the guardrail matched only a fixed phrase list (it had "exact answers" but not
"exam answers"), so small rewordings slipped through.

**Decision.** Replaced the phrase list with token-based detection in the deterministic classifier:
an answers/solutions noun **and** a graded-work token (test/exam/quiz/homework/essay/assignment)
**and** an acquisition phrase (give me / tell me the / answers to / …), OR explicit cheating
(copy/cheat/plagiarise + graded), OR standalone markers (do my homework, write my essay, answer
key). Calibrated so legitimate prep ("how do I answer exam questions", "help me prepare for my
exam") is *not* misclassified.

**Consequences.** All tested integrity rewordings are refused with no tool calls; the eval's
integrity case and legitimate test-prep queries are unaffected.

---

## D-23 — Feedback recognises material-id + sentiment, and a wider sentiment vocabulary (QA it.2)

**Context.** "the M101 material was great" routed to `study_plan`, not `feedback` — the positive
vocabulary was limited to helped/useful/worked and missed common phrasings.

**Decision.** Broadened POSITIVE_FB / NEGATIVE_FB with feedback-context phrases ("was great",
"really helped", "was useless", …) and added an anchor: when a material id (M###) co-occurs with a
bare sentiment adjective, classify as feedback. Kept the generic-word guard so "what's a good way
to study" and "I want to do well" stay non-feedback.

**Consequences.** Material- and topic-level feedback is captured across natural phrasings and
correctly moves the priority score, with no false positives on planning queries.

---

## D-24 — Student ids are canonicalised (whitespace + case) at the boundary (QA it.2)

**Context.** " S123 ", "s123", and "S123\n" all resolved to student-not-found, because lookup was
exact-match. A trailing newline or wrong case shouldn't break resolution, and normalising only the
data lookup would split memory/logs across variants.

**Decision.** Added `Repo.canonical_id()` (strip + case-insensitive resolution against real ids and
duplicate ids, returning the stripped input if unmatched) and routed `get_student` through it. The
pipeline canonicalises `student_id` once at `respond()` entry, and the CLI resolves it right after
constructing the orchestrator, so data, memory, and the session log all key off the same canonical
id. The not-found path is preserved for genuinely unknown ids.

**Consequences.** Whitespace/case variants resolve to the right student with consistent memory and
logging; the duplicate-id integrity refusal (SYN-05) still fires.

---

## D-25 — `top_k` is bounded (QA it.3)

**Context.** `top_k = top_k or default` treated 0 as "use default" and a negative value produced an
empty slice; a huge value was passed straight to the vector store.

**Decision.** `MaterialRetriever.recommend` now takes `k = top_k if int>0 else default`, capped at
50. Degenerate values become a sane bounded request instead of surprising or empty results.

**Consequences.** Retrieval is robust to any `top_k` an LLM might emit; normal calls (top_k=1 from
composition) are unchanged.

---

## D-26 — The CLI turns backend-start failures into a clear message (QA it.5)

**Context.** With `llm.backend: ollama` forced and no server reachable, the CLI exited with a raw
multi-line `ConnectionError` traceback.

**Decision.** `cli.main` wraps orchestrator construction; on failure it prints a one-line,
actionable message (naming the configured backend and the underlying error, and pointing to
`auto`/`deterministic` or starting Ollama) and returns exit code 2. The offline `auto` path is
unaffected.

**Consequences.** A misconfigured or unreachable backend is a clean, understandable error rather
than a stack trace; scripted callers can distinguish it via the exit code.

---

## D-27 — Repositioned as a three-mode experimental comparison, not a deterministic-first product

**Context.** D-01 shipped a deterministic engine as *the* default, with LLM backends as an
opportunistic upgrade when reachable. A dedicated evaluation (`eval/phase1_cases.jsonl`, 67
natural-language cases across 9 categories: paraphrase, guardrail false-positives, multi-turn,
multi-intent, feedback, data edge cases, adversarial true-positives) tested that assumption
directly by running the same cases through the deterministic engine and the existing
`llm.backend: auto` hybrid path (deterministic router first, LLM tool choices unioned on top).
Result: hybrid did not out-score deterministic (~48% weighted correctness either way) because
the deterministic pre-router still gates the majority of cases where an LLM would help —
paraphrase, conversational follow-ups, and guardrail false-positives — and it introduced two
hallucinations deterministic cannot produce by construction, at roughly 1,000× the latency.
Full findings: [`PHASE1_REPORT.md`](PHASE1_REPORT.md).

**Decision.** Two changes. (1) Built a third mode, `llm` (`src/llm/agentic.py`): native
multi-round LLM tool-calling with no deterministic pre-router at all — only structural
validation (tool/arg whitelisting, forced `student_id`) and post-hoc grounding verification
remain as safety nets. (2) Made the mode a first-class, config-only selection
(`pipeline.mode: deterministic | hybrid | llm`, `src/utils/config.py:resolve_mode`, env override
`AUREN_MODE`) rather than an implicit fallback chain, and repositioned the README/DECISIONS
narrative accordingly: this repository evaluates three approaches side by side rather than
prescribing an offline-first default with opportunistic upgrades.

**Consequences.** The `llm` mode recovered most of the coverage gap (72.4% weighted score,
wins or ties in 87% of the 67 cases — see [`PHASE2_REPORT.md`](PHASE2_REPORT.md)) but is not a
strict improvement: it produced 4 confirmed hallucinations (a fabricated test name, a
fabricated material title, a topic misattribution shared with hybrid, and two "claims an action
happened but never called the tool" cases) and, critically, **failed to refuse a direct
system-prompt-extraction jailbreak** that deterministic and hybrid catch 100% of the time by
construction — trading some adversarial robustness for natural-language coverage. This is not
treated as a solved problem: `PHASE2_REPORT.md` §8 gives a scoped four-item recommendation
(grounding that catches prose-level fabrication, not just ids/dates/numbers; a narrow
deterministic safety net for unambiguous adversarial patterns running *after* tool selection
rather than gating it beforehand; a tool-execution invariant; and a shared `material_id → topic`
resolution path) rather than declaring one mode the unconditional winner. Deterministic and
hybrid remain fully supported, auditable options for contexts that value the
safety-by-construction guarantee over coverage.

---

## D-28 — Tests reorganized by mode; a real silent-live-Ollama bug fixed in the process

**Context.** D-27's three-mode pivot needed matching test structure — until this decision the
suite was a flat `tests/*.py` with no way to tell, from a filename alone, which mode(s) a test
exercised, and (per D-20's update note above) its claimed backend pin had silently regressed:
`tests/conftest.py`'s `orch` fixture built an `Orchestrator` from `load_config()` with **no**
mode/backend override, so any test calling `.respond()` picked up whatever `pipeline.mode`/
`llm.backend` happened to be in `config/config.yaml` — which, once D-27 set the default to
`pipeline.mode: hybrid`, meant those tests silently drove **live Ollama calls** whenever a
server was reachable. Measured directly: one such test took **69.5 seconds** in isolation;
the *entire* pre-fix suite (63–75 tests) took 420–440 seconds for what should have been an
offline, sub-second run.

**Decision.** Reorganized `tests/` into `tests/shared/` (mode-agnostic component tests: tools,
retrieval, mastery formula, context-payload mechanics, `resolve_mode`) plus
`tests/mode_deterministic/`, `tests/mode_hybrid/`, `tests/mode_llm/` — each with its **own**
`conftest.py` that explicitly pins `pipeline.mode`/`llm.backend` for that directory's
`orch`/`config` fixtures, so a test's mode is a property of *where the file lives*, never an
accident of the loaded config or ambient Ollama reachability. `tests/conftest.py` itself now
pins `deterministic` explicitly (restoring, correctly this time, what D-20 originally
described) since `tests/shared/` uses it directly with no override. Every mode-specific test
file was also renamed with an explicit mode prefix (`test_deterministic_*.py`,
`test_hybrid_*.py`, `test_llm_*.py`) after discovering pytest's default import mode collides on
identical basenames across sibling directories with no `__init__.py` (e.g. two different
`test_e2e.py` files raise an `import file mismatch` collection error) — the fix (mode-prefixed,
globally-unique basenames) also directly serves the goal of filenames being unambiguous about
which mode they cover.

New coverage was added per mode rather than just reorganizing existing files:
`test_known_results.py` (deterministic) and `test_hybrid_known_failures.py`/
`test_llm_known_failures.py` pin down each mode's specific, verified weaknesses as regression
tests (the two `known_failures` files use `xfail(strict=True)`, cross-referenced to
`PHASE2_REPORT.md` §8, so a fix flips them to an unexpected pass — the signal to remove the
marker rather than silently losing the regression check); `test_llm_improvements_over_other_modes.py`
replays the specific cases where `llm` mode fixed a documented deterministic/hybrid failure, so
that coverage gain can't silently regress either. Two live, Ollama-gated smoke test files
(`test_hybrid_e2e_live.py`, `test_llm_e2e_live.py`) were added for genuine end-to-end
verification against a real model; they skip cleanly (not fail) when no server is reachable, so
the default `uv run pytest` run never requires Ollama.

**Consequences.** `uv run pytest` dropped from ~420–440s to ~80–200s depending on whether
Ollama is reachable (the two live-smoke files add real model latency only when it is), and is
reproducible either way. 107 tests total: 100 passing outright, 7 `xfail`ed (documented, tracked
regressions). Every replayed hybrid/llm transcript in the `known_failures`/
`improvements_over_other_modes` files uses a scripted fake client (`FakeSelectGenerateClient` /
`FakeRawClient`, defined once per mode's `conftest.py`) rather than a live call, so this
regression coverage is exact and reproducible — it verifies the *pipeline's* handling of a
fixed transcript, not that a live model will say the same thing again (re-running the live
suite via `eval/phase1_runner.py` is how that's checked).
