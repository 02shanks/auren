"""Config access: parse config/config.yaml once, merge optional .env, hand out copies."""

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

_CACHE: dict[str, Any] | None = None
_CACHE_PATH: Path | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return a deep copy of the parsed config (callers may mutate their copy freely)."""
    global _CACHE, _CACHE_PATH
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if _CACHE is None or _CACHE_PATH != cfg_path:
        load_dotenv(REPO_ROOT / ".env", override=False)
        with open(cfg_path, encoding="utf-8") as f:
            _CACHE = yaml.safe_load(f) or {}
        _CACHE_PATH = cfg_path
    return copy.deepcopy(_CACHE)


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


_MODES = ("deterministic", "hybrid", "llm")


def resolve_mode(config: dict[str, Any]) -> str:
    """The active pipeline mode: 'deterministic' | 'hybrid' | 'llm'.

    Selected by config (``pipeline.mode``), overridable via the AUREN_MODE env var.
    When absent, derived from the legacy ``llm.backend`` knob for backwards
    compatibility (deterministic backend -> deterministic mode, else hybrid).
    """
    mode = env("AUREN_MODE") or config.get("pipeline", {}).get("mode")
    if mode:
        mode = str(mode).strip().lower()
        if mode not in _MODES:
            raise ValueError(f"unknown pipeline mode '{mode}' (expected one of {_MODES})")
        return mode
    backend = config.get("llm", {}).get("backend", "auto")
    return "deterministic" if backend == "deterministic" else "hybrid"
