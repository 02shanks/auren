"""Embedding backends behind one interface.

- ``HashingEmbedder`` — deterministic char-trigram + word hashed vectors, L2
  normalized. No model, no network; genuine *lexical* similarity (reordered /
  reworded / substring topic titles score high, unrelated ones low). This is the
  offline default that keeps retrieval runnable and tests meaningful (DECISIONS D-02).
- ``OllamaEmbedder`` — real semantic embeddings via ``nomic-embed-text`` when a
  local Ollama server is available.

Reproducibility note: Python's builtin ``hash()`` is per-process salted, so we
use ``hashlib.blake2b`` for a stable, cross-run hash.
"""

import hashlib
import math
import re
from typing import Protocol

from src.utils.logging_config import get_logger

log = get_logger("embedder")

_WORD_RE = re.compile(r"[0-9a-z\u00c0-\uffff]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class HashingEmbedder:
    """Deterministic lexical embedder (char trigrams + word unigrams -> hashed dims)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = max(16, int(dim))
        self.name = f"hashing-{self.dim}"

    def _features(self, text: str) -> list[str]:
        text = " ".join(text.lower().split())
        padded = f"^{text}$"
        grams = [padded[i : i + 3] for i in range(max(0, len(padded) - 2))]
        words = _WORD_RE.findall(text)
        return grams + [f"#{w}" for w in words]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            h = _stable_hash(feat)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OllamaEmbedder:
    """Real embeddings via a local Ollama server (model: nomic-embed-text by default)."""

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None) -> None:
        import ollama  # imported lazily; only needed on the local-model path

        self.model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()
        # Probe once to learn the dimensionality and fail fast if unreachable.
        probe = self._client.embeddings(model=model, prompt="dimension probe")
        self._vec = list(probe["embedding"])
        self.dim = len(self._vec)
        self.name = f"ollama-{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            resp = self._client.embeddings(model=self.model, prompt=t)
            out.append(list(resp["embedding"]))
        return out


def get_embedder(config: dict) -> Embedder:
    rc = config.get("retrieval", {})
    choice = rc.get("embedder", "auto")
    dim = int(rc.get("hashing_dim", 512))
    model = rc.get("ollama_embed_model", "nomic-embed-text")
    if choice in ("ollama", "auto"):
        try:
            emb = OllamaEmbedder(
                model=model, host=config.get("llm", {}).get("ollama", {}).get("host")
            )
            log.info("using Ollama embedder (%s, dim=%d)", model, emb.dim)
            return emb
        except Exception as exc:  # any failure -> offline fallback
            if choice == "ollama":
                raise
            log.warning(
                "Ollama embedder unavailable (%s); using deterministic hashing embedder", exc
            )
    return HashingEmbedder(dim=dim)


if __name__ == "__main__":
    from src.utils.config import load_config

    config = load_config()
    embedder = get_embedder(config)
    print(f"{embedder.name=}, {embedder.dim=} ")
    sentences = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "I like to eat pizza",
        "I like to eat pasta",
        "I like to study physics",
        "I am weak in physics",
        "Tommorrow I have physics exam",
        "I want to study my weak topics",
    ]
    # embed the sentences and calculate the cosine similarity between the all pairs of sentences
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sentence_embedding_i = embedder.embed([sentences[i]])
            sentence_embedding_j = embedder.embed([sentences[j]])
            cosine_similarity = cosine(sentence_embedding_i[0], sentence_embedding_j[0])
            print(
                f"{sentences[i]} and {sentences[j]} have a cosine similarity of {cosine_similarity}"
            )
