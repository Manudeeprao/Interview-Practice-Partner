"""Retrieval helpers used by the interview nodes."""

from __future__ import annotations

import logging
from typing import List, Optional

from .embeddings import embed_texts
from .vector_store import ResumeVectorStore

logger = logging.getLogger(__name__)


class ResumeRetriever:
    """Build and query resume context for a session."""

    def __init__(self, vector_store: Optional[ResumeVectorStore] = None) -> None:
        self.vector_store = vector_store or ResumeVectorStore()

    def index_resume(self, session_id: str, text: str, metadata: Optional[dict] = None) -> List[dict]:
        from .chunker import split_text_into_chunks

        chunks = split_text_into_chunks(text)
        if not chunks:
            return []

        embeddings = embed_texts(chunks)
        safe_metadata = [{"source": "resume", **(metadata or {})} for _ in chunks]
        self.vector_store.add_documents(chunks, embeddings, session_id, safe_metadata)
        return [{"text": chunk, "metadata": meta} for chunk, meta in zip(chunks, safe_metadata)]

    def retrieve_context(self, session_id: str, role: str, history: Optional[list] = None, top_k: int = 5) -> List[str]:
        if not session_id:
            return []

        history_text = ""
        if history:
            history_text = " ".join(msg.get("content", "") for msg in history if isinstance(msg, dict))

        query = (
            f"Role: {role}\n"
            f"Conversation context: {history_text}\n"
            f"Retrieve the candidate's most relevant resume details such as projects, internships, certifications, "
            f"and role-related skills for use in the next interview question."
        ).strip()
        if not query:
            return []

        results = self.vector_store.query(query, session_id=session_id, top_k=top_k)
        return [item.get("text", "") for item in results if item.get("text")]
