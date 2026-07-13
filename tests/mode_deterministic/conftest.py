"""Fixtures for pipeline.mode == 'deterministic' tests.

`orch` here is explicitly pinned — fast, offline, zero model calls, zero dependence on
whether a local Ollama server happens to be running. This is the mode every canonical/e2e
test in this repo was originally written to assume; the pin makes that assumption load-bearing
instead of accidental (see tests/conftest.py's docstring for why this matters).
"""

import datetime as dt

import pytest

from src.orchestrator.pipeline import Orchestrator
from src.utils.config import load_config

TODAY = dt.date(2026, 7, 5)


def _deterministic_config():
    cfg = load_config()
    cfg.setdefault("pipeline", {})["mode"] = "deterministic"
    cfg.setdefault("llm", {})["backend"] = "deterministic"
    return cfg


@pytest.fixture
def config():
    return _deterministic_config()


@pytest.fixture
def orch(config):
    return Orchestrator(config, dataset="all", today=TODAY)
