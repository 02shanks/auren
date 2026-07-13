"""OpenRouter backend (OpenAI-compatible, opt-in cloud fallback). Uses only the
standard library so it adds no dependency. Sampling-param wiring is unit-tested with mocked
HTTP in tests/mode_hybrid/test_hybrid_openrouter_client_params.py; live calls still require
network access and ``OPENROUTER_API_KEY``, so they are not exercised by the offline suite."""

import json
import time
import urllib.error
import urllib.request

from src.llm import prompts
from src.llm.base import Selection, ToolCall
from src.utils.config import env
from src.utils.logging_config import get_logger

log = get_logger("openrouter")


class OpenRouterClient:
    def __init__(self, config: dict) -> None:
        oc = config.get("llm", {}).get("openrouter", {})
        self.base_url = oc.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        self.model = oc.get("chat_model", "")
        self.max_retries = int(oc.get("max_retries", 3))
        # Deterministic sampling: near-zero temperature + fixed seed.
        self.temperature = float(oc.get("temperature", 0.0))
        self.top_p = float(oc.get("top_p", 1.0))
        self.seed = int(oc.get("seed", 42))
        self.api_key = env("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.name = f"openrouter:{self.model}"

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429 and attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
            except urllib.error.URLError as exc:
                last_exc = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenRouter request failed: {last_exc}")

    def select_tools(self, query: str, student_id: str, tools: list[dict]) -> Selection:
        resp = self._post(
            {
                "model": self.model,
                "messages": prompts.selection_messages(query, student_id),
                "tools": tools,
                "tool_choice": "auto",
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            }
        )
        msg = resp.get("choices", [{}])[0].get("message", {})
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(fn.get("name", ""), args))
        return Selection(tool_calls=calls, intent="llm")

    def generate(self, query: str, context: dict) -> str:
        resp = self._post(
            {
                "model": self.model,
                "messages": prompts.generation_messages(query, context),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            }
        )
        return str(resp.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()

    # ---- raw multi-round chat (agentic ``llm`` pipeline mode) ----
    def chat_raw(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One chat round; backend-neutral return (see OllamaClient.chat_raw)."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = self._post(payload)
        msg = resp.get("choices", [{}])[0].get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args, "id": tc.get("id")})
        return {
            "content": str(msg.get("content") or ""),
            "tool_calls": calls,
            "raw_message": msg,
        }

    @staticmethod
    def tool_result_message(name: str, output: dict, call_id: str | None = None) -> dict:
        return {
            "role": "tool",
            "tool_call_id": call_id or name,
            "content": json.dumps(output, ensure_ascii=False),
        }
