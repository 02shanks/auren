"""Fixtures for pipeline.mode == 'llm' tests (the end-to-end agentic pipeline).

Most tests here construct their own Orchestrator with get_llm_client monkeypatched to a
scripted FakeRawClient (see test_llm_agentic_loop.py's `llm_orchestrator` helper) so they stay
fast and fully offline regardless of whether a local Ollama server is reachable — and so a
documented transcript (a hallucination, a jailbreak-refusal failure, a correctly-handled
paraphrase) replays exactly the same way on every run, which a live model cannot guarantee.

The `orch` fixture below builds a REAL, live-Ollama-backed Orchestrator pinned to
pipeline.mode='llm'. It is used only by test_llm_e2e_live.py, which skips itself cleanly if no
Ollama server is reachable — see that file for the reachability check. Note that constructing
an `orch` here with no reachable Ollama raises RuntimeError immediately (AgenticExecutor
requires a raw-chat-capable backend) rather than silently falling back — that fail-fast
behavior is itself covered by test_llm_agentic_loop.py's
test_llm_mode_requires_raw_capable_backend.
"""

import datetime as dt

import pytest

import src.orchestrator.pipeline as pipeline_mod
from src.orchestrator.pipeline import Orchestrator
from src.utils.config import load_config

TODAY = dt.date(2026, 7, 5)


def _llm_config():
    cfg = load_config()
    cfg.setdefault("pipeline", {})["mode"] = "llm"
    cfg.setdefault("llm", {})["backend"] = "ollama"
    return cfg


@pytest.fixture
def config():
    return _llm_config()


@pytest.fixture
def orch(config):
    return Orchestrator(config, dataset="all", today=TODAY)


# --------------------------------------------------------------------------- #
# Shared scripted fake — imported by sibling test files as                    #
# `from conftest import FakeRawClient, llm_orchestrator`                      #
# (pytest's default import mode puts this directory on sys.path, so a plain   #
# module-level import of the conftest works from files that live beside it).  #
# --------------------------------------------------------------------------- #
class FakeRawClient:
    """Scripted raw-chat backend: pops one reply per chat_raw call.

    Replaying a canned response tests the PIPELINE's handling of that transcript (tool
    execution, structural validation, grounding verdict, memory writes) — it does not
    re-verify live model behavior. To re-run a case against the real model, use
    `uv run python -m eval.phase1_runner --mode llm` (see eval/phase1_cases.jsonl).
    """

    name = "fake:scripted"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # (messages, tools) per round, for assertions

    def chat_raw(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        if not self.script:
            return {"content": "…", "tool_calls": [], "raw_message": {"role": "assistant"}}
        step = self.script.pop(0)
        return {
            "content": step.get("content", ""),
            "tool_calls": step.get("tool_calls", []),
            "raw_message": {"role": "assistant", "content": step.get("content", "")},
        }

    @staticmethod
    def tool_result_message(name, output, call_id=None):
        return {"role": "tool", "tool_name": name, "content": str(output)}

    # Selection-protocol stubs (unused in llm mode but keep the interface whole)
    def select_tools(self, query, student_id, tools):  # pragma: no cover
        raise AssertionError("select_tools must not be called in llm mode")

    def generate(self, query, context):  # pragma: no cover
        raise AssertionError("generate must not be called in llm mode")


def llm_orchestrator(monkeypatch, config, script):
    """Build an Orchestrator pinned to pipeline.mode='llm' with get_llm_client
    monkeypatched to a FakeRawClient replaying `script`. Returns (orch, fake)."""
    fake = FakeRawClient(script)
    monkeypatch.setattr(pipeline_mod, "get_llm_client", lambda *a, **k: fake)
    config.setdefault("pipeline", {})["mode"] = "llm"
    return Orchestrator(config, dataset="all", today=TODAY), fake
