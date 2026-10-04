"""ChromaDB-backed vector store for per-session resume chunks."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .embeddings import embed_texts

logger = logging.getLogger(__name__)

_CHROMA_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db"))


class ResumeVectorStore:
    """Simple wrapper around a local ChromaDB collection."""

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        self.persist_directory = persist_directory or _CHROMA_DIR
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="resume_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[str], embeddings: List[List[float]], session_id: str, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if not documents:
            return
        if metadata is None:
            metadata = [{} for _ in documents]

        ids = [f"{session_id}:{idx}" for idx in range(len(documents))]
        payload_metadata = []
        for index, item in enumerate(metadata):
            doc_meta = dict(item or {})
            doc_meta["session_id"] = session_id
            payload_metadata.append(doc_meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=payload_metadata,
        )

    def delete_session(self, session_id: str) -> None:
        try:
            self.collection.delete(where={"session_id": session_id})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to delete session %s from ChromaDB: %s", session_id, exc)

    def query(self, query_text: str, session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query_text.strip():
            return []
        try:
            query_embeddings = embed_texts([query_text])
            if not query_embeddings:
                return []
            n_results = max(1, top_k)
            try:
                count = self.collection.count()
                if count == 0:
                    return []
                n_results = min(n_results, count)
            except Exception:
                pass
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where={"session_id": session_id},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Resume vector query failed for session %s: %s", session_id, exc)
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        output = []
        for doc, meta, distance in zip(docs, metas, distances):
            output.append({"text": doc, "metadata": meta or {}, "distance": distance})
        return output

    def close(self) -> None:
        """Release the persistent Chroma client, especially on Windows."""
        close = getattr(self.client, "close", None)
        if close:
            close()
