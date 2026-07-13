"""Mock-based integration test for the Ollama client's deterministic *sampling* params (not
to be confused with pipeline.mode == 'deterministic' — this file tests the OllamaClient class
that pipeline.mode == 'hybrid' uses for tool selection and generation).

We stub ollama.Client so the REAL OllamaClient code path runs (param assembly,
response parsing) without a live Ollama server. Verifies options + think are
forwarded on both select_tools and generate.
"""

import sys
import types

import pytest

from src.llm.ollama_client import OllamaClient
from src.utils.config import load_config


def _install_fake_ollama():
    """Inject a fake `ollama` module with a recording Client."""
    captured = {"chat": []}

    class _Msg:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class _Client:
        def __init__(self, host=None):
            self.host = host

        def list(self):
            return {}

        def chat(self, **kwargs):
            captured["chat"].append(kwargs)
            # shape the response to the call type
            if "tools" in kwargs:
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weak_topics",
                                    "arguments": {"student_id": "S01"},
                                }
                            }
                        ],
                    }
                }
            return {"message": {"content": "Focus on Algebra (M101)."}}

    fake = types.ModuleType("ollama")
    fake.Client = _Client
    sys.modules["ollama"] = fake
    return captured


def test_ollama_forwards_deterministic_options():
    captured = _install_fake_ollama()
    config = load_config()
    oc = config["llm"]["ollama"]
    client = OllamaClient(config)

    # Client attributes must round-trip from config (no hard-coded literals, so a
    # config value change never breaks this test — it only proves the wiring).
    assert client.temperature == oc["temperature"]
    assert client.top_p == oc["top_p"]
    assert client.top_k == oc["top_k"]
    assert client.seed == oc["seed"]
    assert client.think == oc["think"]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weak_topics",
                "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}},
            },
        }
    ]
    sel = client.select_tools("What should I study?", "S01", tools)
    gen = client.generate("Study plan?", {"student_id": "S01", "tool_outputs": []})

    assert sel.tool_calls[0].name == "get_weak_topics"
    assert gen == "Focus on Algebra (M101)."

    # Both calls must forward the config-derived sampling params + think.
    assert len(captured["chat"]) == 2
    for call in captured["chat"]:
        opts = call["options"]
        assert opts["temperature"] == oc["temperature"]
        assert opts["top_p"] == oc["top_p"]
        assert opts["top_k"] == oc["top_k"]
        assert opts["repeat_penalty"] == oc["repeat_penalty"]
        assert opts["seed"] == oc["seed"]
        assert call["think"] == oc["think"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
