"""ChromaDB-backed vector store for per-session resume chunks.

One persistent client per instance. Every document carries ``session_id`` in
its metadata and every query filters on it, so one session's resume can never
leak into another session's interview.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .embeddings import embed_texts

logger = logging.getLogger(__name__)

_CHROMA_DIR = os.getenv(
    "CHROMA_DB_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db"),
)
_COLLECTION_NAME = "resume_chunks"


class ResumeVectorStore:
    """Thin wrapper around a local ChromaDB collection."""

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        self.persist_directory = persist_directory or _CHROMA_DIR
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_resume(self, session_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Replace a session's resume chunks (delete-then-add; idempotent).

        ``chunks`` are ``{"text": str, "metadata": {...}}`` dicts as produced
        by :func:`rag.chunker.chunk_resume`. Chunk IDs are deterministic
        (``"{session_id}:{chunk_index}"``) so re-uploads never duplicate.
        Returns the number of chunks stored.
        """
        if not session_id:
            raise ValueError("session_id is required to store resume chunks")
        # Delete first so re-uploads replace rather than append.
        self.delete_session(session_id)

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for position, chunk in enumerate(chunks):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            meta = dict(chunk.get("metadata") or {})
            chunk_index = meta.get("chunk_index", position)
            meta["session_id"] = session_id
            meta["chunk_index"] = chunk_index
            documents.append(text)
            metadatas.append(meta)
            ids.append(f"{session_id}:{chunk_index}")

        if not documents:
            logger.warning("add_resume: no non-empty chunks for session %s", session_id)
            return 0

        embeddings = embed_texts(documents)
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Stored %d resume chunks for session %s", len(documents), session_id)
        return len(documents)

    def delete_session(self, session_id: str) -> None:
        """Remove every chunk belonging to a session."""
        if not session_id:
            return
        try:
            self.collection.delete(where={"session_id": session_id})
            logger.info("Deleted ChromaDB chunks for session %s", session_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to delete session %s from ChromaDB: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def query(
        self, session_id: str, query_text: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` chunks for a session, always session-filtered.

        Each result is ``{"text", "metadata", "distance"}`` (cosine distance;
        lower is more similar). Returns [] when the session has no resume.
        """
        if not session_id or not (query_text or "").strip():
            return []
        try:
            query_embeddings = embed_texts([query_text])
            if not query_embeddings:
                return []
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=max(1, k),
                where={"session_id": session_id},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Resume vector query failed for session %s: %s", session_id, exc)
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"text": doc, "metadata": meta or {}, "distance": distance}
            for doc, meta, distance in zip(docs, metas, distances)
            if doc
        ]

    def count_session(self, session_id: str) -> int:
        """Number of stored chunks for a session (0 when none)."""
        if not session_id:
            return 0
        try:
            # NOTE: collection.count() does not reliably support `where`
            # filters across ChromaDB versions; get() does.
            result = self.collection.get(
                where={"session_id": session_id}, include=[]
            )
            return len(result.get("ids", []))
        except Exception:  # pragma: no cover - defensive
            return 0

    def close(self) -> None:
        """Release the persistent Chroma client, especially on Windows."""
        close = getattr(self.client, "close", None)
        if close:
            close()
