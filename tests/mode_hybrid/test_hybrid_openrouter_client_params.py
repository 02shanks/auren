"""Mock-based integration test for the OpenRouter client's deterministic *sampling* params
(not to be confused with pipeline.mode == 'deterministic' — this file tests the
OpenRouterClient class that pipeline.mode == 'hybrid' uses for tool selection and generation).

We stub urllib.request.urlopen so the REAL OpenRouterClient._post path runs (payload
assembly, retry logic, response parsing) with NO network call. Verifies temperature /
top_p / seed are forwarded on both select_tools and generate.
"""

import json

import pytest

from src.llm.openrouter_client import OpenRouterClient
from src.utils.config import load_config


def _fake_urlopen(payloads):
    """Return a context-manager-like response whose .read() yields a canned JSON body."""

    class _Resp:
        def __init__(self, body):
            self._body = body.encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen(req, timeout=None):
        # capture the outgoing request body
        body = req.data.decode("utf-8")
        payloads.append(json.loads(body))
        if "tools" in payloads[-1]:
            return _Resp(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "get_weak_topics",
                                                "arguments": {"student_id": "S01"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            )
        return _Resp(
            json.dumps({"choices": [{"message": {"content": "Focus on Algebra (M101)."}}]})
        )

    return _urlopen


def test_openrouter_forwards_deterministic_params(monkeypatch):
    config = load_config()
    oc = config["llm"]["openrouter"]
    oc["chat_model"] = "qwen/qwen-2.5-7b-instruct:free"
    captured = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(captured))

    client = OpenRouterClient(config)

    # Client attributes round-trip from config (no hard-coded literals).
    assert client.temperature == oc["temperature"]
    assert client.top_p == oc["top_p"]
    assert client.seed == oc["seed"]

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

    # Both calls must forward the config-derived sampling params.
    assert len(captured) == 2
    for p in captured:
        assert p["temperature"] == oc["temperature"]
        assert p["top_p"] == oc["top_p"]
        assert p["seed"] == oc["seed"]
        assert p["model"] == oc["chat_model"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
