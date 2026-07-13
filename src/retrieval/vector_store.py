"""Vector store behind one interface.

- ``ChromaVectorStore`` — persistent ChromaDB collection (cosine space). We supply
  our own precomputed embeddings, so no embedding model runs inside Chroma.
- ``InMemoryVectorStore`` — pure-python cosine store persisted to a small JSON file,
  used when ChromaDB is not installed. Keeps the whole pipeline runnable with zero
  heavy dependencies (DECISIONS D-03/D-10).

Both persist under ``vector_store/`` so ``build_index.py`` writes once and the CLI
reads across process runs.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from src.retrieval.embedder import cosine
from src.utils.config import repo_path
from src.utils.logging_config import get_logger

log = get_logger("vector_store")

Hit = tuple[str, float, dict[str, Any], str]  # (id, similarity, metadata, document)


class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None: ...

    def query(
        self, embedding: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> list[Hit]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    return all(meta.get(k) == v for k, v in where.items())


class InMemoryVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def upsert(self, ids, embeddings, metadatas, documents) -> None:
        for i, emb, meta, doc in zip(ids, embeddings, metadatas, documents, strict=True):
            self._data[i] = {"embedding": emb, "metadata": meta, "document": doc}
        self._save()

    def query(self, embedding, top_k, where=None) -> list[Hit]:
        scored: list[Hit] = []
        for i, rec in self._data.items():
            if not _matches(rec["metadata"], where):
                continue
            sim = cosine(embedding, rec["embedding"])
            scored.append((i, sim, rec["metadata"], rec["document"]))
        scored.sort(key=lambda h: h[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data = {}
        self._save()


class ChromaVectorStore:
    def __init__(self, path: Path, collection: str) -> None:
        import chromadb  # lazy import; only on the --extra vector path

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, embeddings, metadatas, documents) -> None:
        self._col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, embedding, top_k, where=None) -> list[Hit]:
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where or None,
            include=["metadatas", "documents", "distances"],
        )
        hits: list[Hit] = []
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        for i, dist, meta, doc in zip(ids, dists, metas, docs, strict=False):
            hits.append((i, 1.0 - float(dist), meta or {}, doc or ""))
        return hits

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        # Recreate the collection to drop all vectors.
        name = self._col.name
        self._client.delete_collection(name)
        self._col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


def get_vector_store(config: dict) -> VectorStore:
    rc = config.get("retrieval", {})
    choice = rc.get("vector_store", "auto")
    path = repo_path(rc.get("chroma_path", "vector_store"))
    collection = rc.get("collection", "study_materials")
    if choice in ("chroma", "auto"):
        try:
            store = ChromaVectorStore(Path(path), collection)
            log.info("using ChromaDB vector store at %s", path)
            return store
        except Exception as exc:  # any import/runtime failure -> pure-python fallback
            if choice == "chroma":
                raise
            log.warning("ChromaDB unavailable (%s); using in-memory vector store", exc)
    return InMemoryVectorStore(Path(path) / "inmem_index.json")
