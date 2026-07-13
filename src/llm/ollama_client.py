"""Local Ollama backend (function-calling). Used when reachable; otherwise the
factory falls back to the deterministic engine. Backs both ``hybrid`` mode
(select_tools/generate) and ``llm`` mode (chat_raw's agentic loop). Sampling-param wiring is
unit-tested against a stubbed ``ollama.Client`` in tests/mode_hybrid/test_hybrid_ollama_client_params.py;
live behavior is exercised by tests/mode_hybrid/test_hybrid_e2e_live.py and
tests/mode_llm/test_llm_e2e_live.py, both of which skip cleanly if no Ollama server is reachable."""

import json
from typing import Any

from src.llm import prompts
from src.llm.base import Selection, ToolCall
from src.utils.config import env


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OllamaClient:
    def __init__(self, config: dict) -> None:
        import ollama  # lazy: only needed on this path

        oc = config.get("llm", {}).get("ollama", {})
        self.model = oc.get("chat_model", "qwen3:8b")
        host = env("OLLAMA_HOST") or oc.get("host")
        # Deterministic sampling: near-zero temperature + fixed seed.
        self.temperature = float(oc.get("temperature", 0.0))
        self.top_p = float(oc.get("top_p", 1.0))
        self.top_k = int(oc.get("top_k", 1))
        self.repeat_penalty = float(oc.get("repeat_penalty", 1.1))
        self.seed = int(oc.get("seed", 42))
        self.think = oc.get("think", False)
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._client.list()  # fail fast if the server is unreachable
        self.name = f"ollama:{self.model}"

    def _options(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
        }

    def select_tools(self, query: str, student_id: str, tools: list[dict]) -> Selection:
        resp = self._client.chat(
            model=self.model,
            messages=prompts.selection_messages(query, student_id),
            tools=tools,
            options=self._options(),
            think=self.think,
        )
        msg = _get(resp, "message", {})
        calls: list[ToolCall] = []
        for tc in _get(msg, "tool_calls", []) or []:
            fn = _get(tc, "function", {})
            args = _get(fn, "arguments", {}) or {}
            calls.append(ToolCall(_get(fn, "name", ""), dict(args)))
        return Selection(tool_calls=calls, intent="llm")

    def generate(self, query: str, context: dict) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=prompts.generation_messages(query, context),
            options=self._options(),
            think=self.think,
        )
        return str(_get(_get(resp, "message", {}), "content", "")).strip()

    # ---- raw multi-round chat (agentic ``llm`` pipeline mode) ----
    def chat_raw(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One chat round. Returns a backend-neutral dict:
        {content, tool_calls: [{name, arguments, id}], raw_message} where raw_message
        is appendable to the messages list to continue the conversation."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "options": self._options(),
            "think": self.think,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat(**kwargs)
        msg = _get(resp, "message", {})
        calls = []
        for tc in _get(msg, "tool_calls", []) or []:
            fn = _get(tc, "function", {})
            calls.append(
                {
                    "name": _get(fn, "name", ""),
                    "arguments": dict(_get(fn, "arguments", {}) or {}),
                    "id": None,  # ollama tool calls carry no ids
                }
            )
        return {
            "content": str(_get(msg, "content", "") or ""),
            "tool_calls": calls,
            "raw_message": msg,
        }

    @staticmethod
    def tool_result_message(name: str, output: dict, call_id: str | None = None) -> dict:
        return {
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(output, ensure_ascii=False),
        }


if __name__ == "__main__":
    from src.utils.config import load_config

    config = load_config()
    client = OllamaClient(config)
    print(client.generate("What is the capital of France?", {}))
