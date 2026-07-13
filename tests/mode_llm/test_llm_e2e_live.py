"""pipeline.mode == 'llm' — REAL, live-Ollama-backed end-to-end smoke test. Skips itself
cleanly if no Ollama server is reachable, so the offline suite (test_llm_agentic_loop.py etc.,
which use a scripted FakeRawClient) is what CI relies on; this file is for a human — or an
environment with Ollama available — to confirm the agentic tool-calling loop actually works
against the real model configured in config/config.yaml (default qwen3:8b).

Note: llm mode is slower than hybrid per turn (a separate model call per tool-calling round;
see PHASE2_REPORT.md §7 for measured latency), so this file uses a longer per-test budget and
keeps to one or two queries rather than a full sweep — for the full 67-case sweep, use
`uv run python -m eval.phase1_runner --mode llm`.

Run explicitly:  uv run pytest tests/mode_llm/test_llm_e2e_live.py -v
"""

import datetime as dt

import pytest
import requests

from src.orchestrator.pipeline import Orchestrator


def _ollama_reachable(host: str) -> bool:
    try:
        return requests.get(f"{host.rstrip('/')}/api/tags", timeout=3).status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture
def live_orch(config):
    host = config.get("llm", {}).get("ollama", {}).get("host", "http://localhost:11434")
    if not _ollama_reachable(host):
        pytest.skip(f"no Ollama server reachable at {host}; skipping live llm-mode smoke test")
    return Orchestrator(config, dataset="all", today=dt.date(2026, 7, 5))


def test_live_canonical_query_answers_grounded(live_orch, reset_memory):
    reset_memory("S123")
    r = live_orch.respond(
        "I am weak in Algebra, what should I study next?", "S123", live_orch.new_session()
    )
    assert r["answer"].strip()
    assert "get_weak_topics" in r["tools_called"]


def test_live_paraphrase_not_declined(live_orch, reset_memory):
    """The one behavior this mode exists to fix (PHASE1_REPORT.md §3.1): a paraphrased study
    question with no deterministic trigger keyword must reach the model and get a real answer,
    not the generic off_topic decline."""
    reset_memory("S123")
    r = live_orch.respond(
        "I keep messing up trigonometry sums", "S123", live_orch.new_session()
    )
    assert r["intent"] != "off_topic"
    assert r["answer"].strip()
