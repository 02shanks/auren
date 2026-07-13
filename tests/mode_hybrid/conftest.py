"""Fixtures for pipeline.mode == 'hybrid' tests.

Hybrid still runs the deterministic classifier first (it can short-circuit before any model
is consulted, and its tool choices are unioned with the LLM's — see src/orchestrator/router.py).
Most tests here therefore don't need a live model at all: they either prove the deterministic
pre-router's short-circuit still holds (a fake client that raises if ever called), or replay a
scripted fake backend to regression-test a documented transcript (see test_hybrid_known_failures.py).

The `orch` fixture below builds a REAL, live-Ollama-backed Orchestrator pinned to
pipeline.mode='hybrid'. It is used only by test_hybrid_e2e_live.py, which skips itself cleanly if no
Ollama server is reachable — see that file for the reachability check.
"""

import datetime as dt

import pytest

from src.llm.base import Selection
from src.orchestrator.pipeline import Orchestrator
from src.utils.config import load_config

TODAY = dt.date(2026, 7, 5)


def _hybrid_config():
    cfg = load_config()
    cfg.setdefault("pipeline", {})["mode"] = "hybrid"
    cfg.setdefault("llm", {})["backend"] = "ollama"
    return cfg


@pytest.fixture
def config():
    return _hybrid_config()


@pytest.fixture
def orch(config):
    return Orchestrator(config, dataset="all", today=TODAY)


# --------------------------------------------------------------------------- #
# Shared scripted fake — imported by sibling test files as                    #
# `from conftest import FakeSelectGenerateClient, hybrid_orchestrator`        #
# --------------------------------------------------------------------------- #
class FakeSelectGenerateClient:
    """Scripted select_tools/generate backend matching src/llm/base.py's LLMClient
    protocol — the interface hybrid mode's router/pipeline actually calls (NOT the raw
    chat_raw loop llm mode uses).

    Replaying a canned response tests the PIPELINE's handling of that transcript — it does
    not re-verify live model behavior. To re-run a case against the real model, use
    `uv run python -m eval.phase1_runner --mode hybrid` (see eval/phase1_cases.jsonl).
    """

    name = "fake:hybrid-scripted"

    def __init__(self, generate_text="Focus on Algebra first (M101).", extra_tool_calls=None):
        self.generate_text = generate_text
        self.extra_tool_calls = extra_tool_calls or []
        self.select_calls = []
        self.generate_calls = []

    def select_tools(self, query, student_id, tools):
        self.select_calls.append(query)
        return Selection(tool_calls=list(self.extra_tool_calls), intent="llm")

    def generate(self, query, context):
        self.generate_calls.append((query, context))
        return self.generate_text


class RaisingClient:
    """A client that fails the test loudly if the LLM is ever consulted — used to prove the
    deterministic pre-router's guardrail short-circuit means select_tools/generate are never
    reached for an unambiguous adversarial case."""

    name = "fake:must-not-be-called"

    def select_tools(self, query, student_id, tools):
        raise AssertionError("select_tools must not be called for a guardrail short-circuit")

    def generate(self, query, context):
        raise AssertionError("generate must not be called for a guardrail short-circuit")


def hybrid_orchestrator(monkeypatch, config, fake):
    """Build an Orchestrator pinned to pipeline.mode='hybrid' with get_llm_client
    monkeypatched to `fake` (a FakeSelectGenerateClient or RaisingClient)."""
    import src.orchestrator.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_llm_client", lambda *a, **k: fake)
    config.setdefault("pipeline", {})["mode"] = "hybrid"
    return Orchestrator(config, dataset="all", today=TODAY)
