"""Auren CLI — a thin shell over the orchestrator.

No business logic lives here: it parses args, runs a REPL, and prints answers to
stdout while all logs go to stderr. Commands: a plain line is a question;
``feedback <topic> helped|not_helped`` logs feedback; ``reset-memory`` clears this
student's memory; ``exit``/``quit`` ends the session (runs reflection first).
"""

import argparse
import sys

from src.memory.store import MemoryStore
from src.orchestrator.pipeline import Orchestrator
from src.utils.config import env, load_config, resolve_mode
from src.utils.logging_config import setup_logging, start_session_log, stop_session_log

BANNER = "Auren — your study assistant. Ask a question, or type 'exit' to finish."


def _handle(orch: Orchestrator, session, student_id: str, line: str) -> bool:
    """Return False when the session should end."""
    stripped = line.strip()
    if not stripped:
        return True
    low = stripped.lower()

    if low in ("exit", "quit"):
        summary, _persona = orch.finalize_session(session, student_id)
        print(f"\nSession ended. Progress saved.\n({summary})")
        return False

    if low == "reset-memory":
        MemoryStore(student_id, orch.config).reset()
        session.mastery_cache.pop(student_id, None)
        print("Memory cleared for this student.")
        return True

    if low.startswith("feedback "):
        parts = stripped.split()
        if len(parts) >= 3:
            signal = parts[-1].lower()
            topic = " ".join(parts[1:-1])
            verb = "helped" if signal.startswith("help") else "did not help"
            query = f"the {topic} material {verb}"
        else:
            print("Usage: feedback <topic> helped|not_helped")
            return True
        result = orch.respond(query, student_id, session)
        print(result["answer"])
        return True

    result = orch.respond(stripped, student_id, session)
    print(result["answer"])
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="auren", description="Auren study assistant CLI")
    parser.add_argument("--student-id", required=True, help="Active student id (e.g. S123)")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["sample", "synthetic", "all"],
        help="Which dataset to load (default: all)",
    )
    parser.add_argument("--once", metavar="QUERY", help="Answer a single query and exit")
    parser.add_argument(
        "--reset-memory", action="store_true", help="Clear this student's memory before starting"
    )
    args = parser.parse_args(argv)

    setup_logging(env("AUREN_LOG_LEVEL") or "WARNING")
    config = load_config()

    try:
        orch = Orchestrator(config, dataset=args.dataset)
    except Exception as exc:
        mode = resolve_mode(config)
        backend = config.get("llm", {}).get("backend", "auto")
        print(
            f"Auren couldn't start with pipeline.mode='{mode}', llm.backend='{backend}': {exc}\n"
            "Start your Ollama server, or set pipeline.mode to 'deterministic' in "
            "config/config.yaml (or the AUREN_MODE env var) to use the offline engine. "
            "Note pipeline.mode='llm' requires llm.backend to be 'ollama' or 'openrouter' — "
            "'deterministic' is not a valid backend for that mode.",
            file=sys.stderr,
        )
        return 2
    student_id = orch.repo.canonical_id(args.student_id)  # tolerate whitespace/case

    if args.reset_memory:
        MemoryStore(student_id, config).reset()

    session = orch.new_session()
    log_dir, log_handler = start_session_log(session.session_id)
    print(f"[session {session.session_id}] trace log: {log_dir / 'session.log'}", file=sys.stderr)

    try:
        if args.once:
            result = orch.respond(args.once, student_id, session)
            print(result["answer"])
            orch.finalize_session(session, student_id)
            return 0

        print(BANNER)
        while True:
            try:
                line = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                orch.finalize_session(session, student_id)
                print("\nSession ended. Progress saved.")
                break
            if not _handle(orch, session, student_id, line):
                break
        return 0
    finally:
        stop_session_log(log_handler)


if __name__ == "__main__":
    sys.exit(main())
