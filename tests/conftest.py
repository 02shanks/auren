"""Shared, root-level fixtures for tests/shared/ (mode-agnostic component tests: tools,
retrieval, mastery scoring, context-payload mechanics — none of these call `orch.respond()`).

The `config`/`orch` fixtures here are pinned to **pipeline.mode: deterministic** explicitly,
not derived from whatever `config/config.yaml` or a live Ollama's reachability happens to
resolve to. This matters: before this pin existed, `orch` silently picked up the ambient
`llm.backend`/`pipeline.mode` default, and any test elsewhere that called `orch.respond()`
was — undetected — driving a real ~30-70s Ollama call per assertion whenever a local Ollama
server happened to be running (confirmed: a single such test took 69.5s in isolation). Pinning
here means tests/shared/'s `orch` fixture is always fast and network-free, matching what a
"component-level" test should be regardless of backend.

Every mode-specific behavior test lives under tests/mode_deterministic/, tests/mode_hybrid/,
or tests/mode_llm/, each with its own conftest.py that overrides `config`/`orch` to pin that
directory's mode explicitly — so which mode a test exercises is a property of *where the file
lives*, never an accident of ambient config or a locally-running Ollama server.
"""

import datetime as dt
import shutil

import pytest

from src.orchestrator.pipeline import Orchestrator
from src.utils.config import load_config, repo_path

TODAY = dt.date(2026, 7, 5)


def _deterministic_config():
    cfg = load_config()
    cfg.setdefault("pipeline", {})["mode"] = "deterministic"
    cfg.setdefault("llm", {})["backend"] = "deterministic"
    return cfg


@pytest.fixture(scope="session")
def config():
    return _deterministic_config()


@pytest.fixture
def reset_memory(config):
    def _reset(*student_ids: str) -> None:
        for sid in student_ids:
            d = repo_path(config["memory"]["root"]) / sid
            if d.exists():
                shutil.rmtree(d)

    return _reset


@pytest.fixture
def orch(config):
    return Orchestrator(config, dataset="all", today=TODAY)
