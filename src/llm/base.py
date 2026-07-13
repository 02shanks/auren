"""LLM backend abstraction.

A deliberately *domain-shaped* interface rather than a raw chat wrapper, so the
``pipeline.mode: deterministic | hybrid`` orchestrator control flow is identical across
backends:

    select_tools(query, student_id, tools) -> Selection   # which tools to call this turn
    generate(query, context)              -> str          # grounded final answer

Three backends implement it: the deterministic offline engine (the zero-setup, model-free
option), local Ollama, and cloud OpenRouter. ``get_llm_client`` picks one from config, and
``auto`` prefers Ollama when reachable, else the deterministic engine (DECISIONS D-06).

``pipeline.mode: llm`` needs more than this protocol — its agentic tool-calling loop
(src/llm/agentic.py) requires a ``chat_raw`` method, which only ``OllamaClient`` and
``OpenRouterClient`` implement; the deterministic client cannot back that mode at all
(``AgenticExecutor`` fails fast if given a client without ``chat_raw``).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.utils.logging_config import get_logger

log = get_logger("llm")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Selection:
    tool_calls: list[ToolCall] = field(default_factory=list)
    intent: str = "unknown"
    # For guardrail intents the router answers directly and skips tools.
    direct_answer: str | None = None
    # Extracted hints the orchestrator uses to compose follow-up recommend calls (sec 8.2).
    focus_topic: str | None = None
    focus_subject: str | None = None


class LLMClient(Protocol):
    name: str

    def select_tools(self, query: str, student_id: str, tools: list[dict]) -> Selection: ...

    def generate(self, query: str, context: dict) -> str: ...


def get_llm_client(config: dict, repo: Any = None) -> LLMClient:
    """Return the configured LLM client. ``repo`` is used only by the deterministic engine."""
    from src.llm.deterministic import DeterministicClient

    backend = config.get("llm", {}).get("backend", "auto")

    if backend in ("ollama", "auto"):
        try:
            from src.llm.ollama_client import OllamaClient

            client = OllamaClient(config)
            log.info("using Ollama LLM backend (%s)", client.name)
            return client
        except Exception as exc:
            if backend == "ollama":
                raise
            log.warning("Ollama LLM unavailable (%s); using deterministic offline engine", exc)

    if backend == "openrouter":
        from src.llm.openrouter_client import OpenRouterClient

        client = OpenRouterClient(config)
        log.info("using OpenRouter LLM backend (%s)", client.name)
        return client

    return DeterministicClient(config, repo=repo)


if __name__ == "__main__":
    from src.utils.config import load_config

    config = load_config()
    client = get_llm_client(config)
    print(client)
