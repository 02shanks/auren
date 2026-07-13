"""Phase-1 NL test-suite runner.

Drives the real Orchestrator over eval/phase1_cases.jsonl in one of the three pipeline modes,
capturing intent, tools, answer, grounding flags, and latency per turn. The run date is pinned
to 2026-07-05 (same as the eval harness) so fixture semantics hold. Each mode writes its own
clearly-named artifact: eval/phase1_results_<mode>.jsonl (deterministic / hybrid / llm).

Run:  uv run python -m eval.phase1_runner --mode deterministic
      uv run python -m eval.phase1_runner --mode hybrid
      uv run python -m eval.phase1_runner --mode llm

After running all three, merge + grade + scorecard:
      uv run python -m eval.mode_scorecard
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import MemoryStore
from src.orchestrator.pipeline import Orchestrator
from src.utils.config import load_config

EVAL_DIR = Path(__file__).resolve().parent
TODAY = dt.date(2026, 7, 5)


# The three canonical, config-selectable pipeline modes -> (pipeline.mode, llm.backend).
CANONICAL_MODES = {
    "deterministic": ("deterministic", "deterministic"),
    "hybrid": ("hybrid", "ollama"),
    "llm": ("llm", "ollama"),
}
# Legacy aliases some earlier scripts/notes used, kept only so an old command line still
# works — always remapped to a canonical name (never their own artifact filename; see
# main()) so "phase1_results_ollama.jsonl"-style ambiguity (was this hybrid or llm? Ollama
# backs both) can't recur.
_LEGACY_ALIASES = {"ollama": "hybrid", "auto": "hybrid"}
MODES = {**CANONICAL_MODES, "ollama": CANONICAL_MODES["hybrid"], "auto": CANONICAL_MODES["hybrid"]}


def run(mode: str, cases_file: str, out_file: str) -> int:
    canonical_mode = _LEGACY_ALIASES.get(mode, mode)
    config = load_config()
    pipeline_mode, backend = MODES[mode]
    config.setdefault("pipeline", {})["mode"] = pipeline_mode
    config.setdefault("llm", {})["backend"] = backend

    orch = Orchestrator(config, dataset="all", today=TODAY)
    print(f"backend in use: {orch.client.name}")

    cases = [
        json.loads(line)
        for line in (EVAL_DIR / cases_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_path = EVAL_DIR / out_file
    n_err = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for case in cases:
            sid = case["student"]
            try:
                MemoryStore(orch.repo.canonical_id(sid), config).reset()
            except Exception:
                pass
            session = orch.new_session()
            turns_out = []
            for q in case["turns"]:
                t0 = time.perf_counter()
                try:
                    r = orch.respond(q, sid, session)
                    turns_out.append(
                        {
                            "query": q,
                            "intent": r["intent"],
                            "tools_called": r["tools_called"],
                            "answer": r["answer"],
                            "grounding_ok": r["grounding_ok"],
                            "grounded_fallback": r["grounded_fallback"],
                            "notes": r["notes"],
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                        }
                    )
                except Exception as exc:
                    n_err += 1
                    turns_out.append(
                        {
                            "query": q,
                            "error": f"{type(exc).__name__}: {exc}",
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                        }
                    )
            rec = {
                "id": case["id"],
                "category": case["category"],
                "student": sid,
                "expect": case["expect"],
                "mode": canonical_mode,
                "turns": turns_out,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{case['id']}] done ({sum(t['latency_ms'] for t in turns_out):.0f} ms)")
    print(f"wrote {out_path} ({len(cases)} cases, {n_err} turn errors)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument("--cases", default="phase1_cases.jsonl")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    canonical = _LEGACY_ALIASES.get(args.mode, args.mode)
    if args.mode in _LEGACY_ALIASES:
        print(f"note: --mode {args.mode!r} is a legacy alias for {canonical!r}; using that name")
    out = args.out or f"phase1_results_{canonical}.jsonl"
    return run(args.mode, args.cases, out)


if __name__ == "__main__":
    sys.exit(main())
