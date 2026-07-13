"""Mode-selection mechanism (mode-agnostic by definition — this is the function that decides
*which* mode's config another test's fixtures pin to): ``resolve_mode`` reads ``pipeline.mode``
(env override ``AUREN_MODE``) or falls back to the legacy ``llm.backend`` knob."""

import pytest

from src.utils.config import resolve_mode


def test_resolve_mode_from_pipeline_key():
    assert resolve_mode({"pipeline": {"mode": "llm"}}) == "llm"
    assert resolve_mode({"pipeline": {"mode": "Deterministic"}}) == "deterministic"


def test_resolve_mode_legacy_backend_fallback():
    assert resolve_mode({"llm": {"backend": "deterministic"}}) == "deterministic"
    assert resolve_mode({"llm": {"backend": "auto"}}) == "hybrid"
    assert resolve_mode({}) == "hybrid"


def test_resolve_mode_env_override(monkeypatch):
    monkeypatch.setenv("AUREN_MODE", "llm")
    assert resolve_mode({"pipeline": {"mode": "deterministic"}}) == "llm"


def test_resolve_mode_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_mode({"pipeline": {"mode": "agentic-ultra"}})
