"""Three-mode comparison scorecard.

Reads eval/phase1_graded.json — one entry per test category, each with per-case verdicts for
all three pipeline modes (deterministic / hybrid / llm) against a documented expected
behavior — and prints a scorecard analogous to eval/run_eval.py's, but comparing modes instead
of scoring a single backend.

This is the third leg of the evaluation pipeline:

    eval/phase1_cases.jsonl          (67 natural-language test cases, 9 categories)
      -> eval/phase1_runner.py --mode {deterministic,hybrid,llm}
           -> eval/phase1_results_<mode>.jsonl   (raw per-case transcripts)
      -> (grading: see PHASE1_REPORT.md / PHASE2_REPORT.md methodology note)
           -> eval/phase1_graded.json            (per-case verdicts, all 3 modes)
      -> eval/mode_scorecard.py                  (this script: the printable comparison)

Unlike run_eval.py, this does NOT fail the process on a low score — there is no "100% or
fail" bar here, because the whole point is comparing three different tradeoffs, not proving
one mode's correctness. Exit code is always 0 unless eval/phase1_graded.json is missing or
malformed.

Run:  uv run python -m eval.mode_scorecard
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
MODES = ("deterministic", "hybrid", "llm")
WEIGHTS = {"correct": 1.0, "partial": 0.5, "wrong_but_safe": 0.0, "unsafe_or_leaked": 0.0}


def _load_graded() -> list[dict]:
    path = EVAL_DIR / "phase1_graded.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run all three modes first "
            "(uv run python -m eval.phase1_runner --mode <deterministic|hybrid|llm>), "
            "then produce eval/phase1_graded.json (see PHASE1_REPORT.md methodology note)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _bar(pct: float, width: int = 20) -> str:
    return "#" * int(round(pct / 100 * width))


def main() -> int:
    graded = _load_graded()

    verdict_counts = {m: Counter() for m in MODES}
    best_mode_counts = Counter()
    best_mode_by_category = defaultdict(Counter)
    unsafe_cases = defaultdict(list)
    total_cases = 0

    for block in graded:
        category = block["category"]
        for case in block["cases"]:
            total_cases += 1
            for mode in MODES:
                verdict = case[mode]["verdict"]
                verdict_counts[mode][verdict] += 1
                if verdict == "unsafe_or_leaked":
                    unsafe_cases[mode].append((category, case["id"], case[mode]["issue"]))
            best_mode_counts[case["best_mode"]] += 1
            best_mode_by_category[category][case["best_mode"]] += 1

    print("=" * 68)
    print("AUREN THREE-MODE COMPARISON SCORECARD")
    print(f"({total_cases} natural-language cases across {len(graded)} categories)")
    print("=" * 68)

    print("\n-- weighted correctness score (correct=1.0, partial=0.5) --")
    for mode in MODES:
        vc = verdict_counts[mode]
        score = sum(WEIGHTS[v] * n for v, n in vc.items())
        pct = 100.0 * score / total_cases if total_cases else 0.0
        detail = ", ".join(f"{v}={vc.get(v, 0)}" for v in WEIGHTS)
        print(f"  {mode:14s} {score:5.1f}/{total_cases}  ({pct:5.1f}%)  {_bar(pct)}")
        print(f"    {detail}")

    print("\n-- unsafe / hallucinated cases --")
    for mode in MODES:
        cases = unsafe_cases[mode]
        print(f"  {mode:14s} {len(cases)} case(s)")
        for category, cid, issue in cases:
            print(f"      [{category}/{cid}] {issue[:100]}")

    print("\n-- best-mode tally (per case, ties/none included) --")
    for label in (*MODES, "tie", "none"):
        print(f"  {label:14s} {best_mode_counts.get(label, 0)}")

    print("\n-- best-mode by category --")
    for category, counts in best_mode_by_category.items():
        parts = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {category:16s} {parts}")

    print("\n" + "-" * 68)
    print("See PHASE1_REPORT.md (deterministic vs hybrid) and PHASE2_REPORT.md")
    print("(full 3-way comparison, tradeoffs, latency, recommendation) for analysis.")
    print("-" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
