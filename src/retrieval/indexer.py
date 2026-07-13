"""Index building + material retrieval (exact match first, semantic fallback second).

Implements blueprint sec 6: structured containment match is the default; the
embedding/vector path fires only when structured match finds nothing, and a
minimum-similarity threshold gates an honest "no match". Only the
study-materials catalog is embedded — the other JSON is looked up, not embedded.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.retrieval.embedder import Embedder, get_embedder
from src.retrieval.vector_store import VectorStore, get_vector_store
from src.utils.config import repo_path
from src.utils.data_loader import Material, load_dataset
from src.utils.logging_config import get_logger

log = get_logger("indexer")

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text.lower())
    return " ".join(text.split())


def _doc_for(m: Material) -> str:
    return f"{m.topic}: {m.title}"


def catalog_hash(materials: list[Material]) -> str:
    payload = json.dumps(
        sorted([[m.material_id, m.topic, m.title] for m in materials]), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_path(config: dict) -> Path:
    return (
        repo_path(config.get("retrieval", {}).get("chroma_path", "vector_store")) / "index_hash.txt"
    )


def build_index(materials: list[Material], config: dict, embedder: Embedder | None = None) -> int:
    """Embed every material and upsert into the vector store keyed by material_id."""
    embedder = embedder or get_embedder(config)
    store = get_vector_store(config)
    store.clear()
    if not materials:
        _hash_path(config).parent.mkdir(parents=True, exist_ok=True)
        _hash_path(config).write_text(catalog_hash(materials), encoding="utf-8")
        return 0
    docs = [_doc_for(m) for m in materials]
    embeddings = embedder.embed(docs)
    metadatas = [
        {
            "topic": m.topic,
            "title": m.title,
            "subject": m.subject or "",
            "material_id": m.material_id,
        }
        for m in materials
    ]
    ids = [m.material_id for m in materials]
    store.upsert(ids, embeddings, metadatas, docs)
    hp = _hash_path(config)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(catalog_hash(materials), encoding="utf-8")
    log.info("indexed %d materials with %s", len(materials), embedder.name)
    return len(materials)


def needs_rebuild(materials: list[Material], config: dict) -> bool:
    hp = _hash_path(config)
    if not hp.exists():
        return True
    try:
        return hp.read_text(encoding="utf-8").strip() != catalog_hash(materials)
    except OSError:
        return True


@dataclass
class MaterialRetriever:
    materials: list[Material]
    store: VectorStore
    embedder: Embedder
    min_similarity: float
    default_top_k: int = 3

    def recommend(self, topic: str, top_k: int | None = None) -> list[dict[str, Any]]:
        # tolerate degenerate top_k (0, negative, non-int, absurdly large) -> sane bounded k
        k = top_k if isinstance(top_k, int) and top_k > 0 else self.default_top_k
        k = min(k, 50)
        q = _normalize(topic or "")
        if not q:
            return []
        # ---- 1. structured / exact-containment match ----
        exact: list[dict[str, Any]] = []
        for m in self.materials:
            nt, ntitle = _normalize(m.topic), _normalize(m.title)
            if q == nt or q == ntitle:
                score = 1.0
            elif q in nt or nt in q or q in ntitle:
                score = 0.85
            else:
                continue
            exact.append(self._row(m, "exact", score))
        if exact:
            exact.sort(key=lambda r: r["score"], reverse=True)
            return exact[:k]
        # ---- 2. semantic fallback (only if structured match found nothing) ----
        vec = self.embedder.embed([topic])[0]
        hits = self.store.query(vec, top_k=k)
        by_id = {m.material_id: m for m in self.materials}
        out: list[dict[str, Any]] = []
        for mid, sim, meta, _doc in hits:
            if sim < self.min_similarity:
                continue
            m = by_id.get(mid)
            if m is None:
                out.append(
                    {
                        "material_id": mid,
                        "title": meta.get("title", ""),
                        "topic": meta.get("topic", ""),
                        "match_type": "semantic",
                        "score": round(sim, 4),
                    }
                )
            else:
                out.append(self._row(m, "semantic", round(sim, 4)))
        return out

    @staticmethod
    def _row(m: Material, match_type: str, score: float) -> dict[str, Any]:
        return {
            "material_id": m.material_id,
            "title": m.title,
            "topic": m.topic,
            "match_type": match_type,
            "score": score,
        }


def get_retriever(
    config: dict, dataset: str = "all", materials: list[Material] | None = None
) -> MaterialRetriever:
    """Build (or reuse) the index for ``dataset`` and return a ready retriever."""
    if materials is None:
        materials = load_dataset(dataset).materials()
    embedder = get_embedder(config)
    if needs_rebuild(materials, config):
        build_index(materials, config, embedder=embedder)
    store = get_vector_store(config)
    rc = config.get("retrieval", {})
    return MaterialRetriever(
        materials=materials,
        store=store,
        embedder=embedder,
        min_similarity=float(rc.get("semantic_min_similarity", 0.60)),
        default_top_k=int(rc.get("default_top_k", 3)),
    )
