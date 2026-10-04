"""Retrieval helpers used by the interview nodes.

The retrieval query is built from the *current* interview context — the role,
the question being asked, and the candidate's last answer — not the whole
conversation history. Results are always scoped to one session, ranked by
cosine distance, and filtered by a relevance threshold.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from .chunker import chunk_resume
from .vector_store import ResumeVectorStore

logger = logging.getLogger(__name__)

_TOP_K_DEFAULT = 5
# Cosine distance on normalized embeddings (distance = 1 - similarity).
# Chunks less similar than this are dropped as irrelevant.
_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.6"))
_DEBUG_RAG = os.getenv("DEBUG_RAG", "0") == "1"


class ResumeRetriever:
    """Build and query resume context for a session."""

    def __init__(self, vector_store: Optional[ResumeVectorStore] = None) -> None:
        self.vector_store = vector_store or ResumeVectorStore()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_resume(
        self, session_id: str, text: str, metadata: Optional[dict] = None
    ) -> List[dict]:
        """Chunk (section-aware), embed and store a resume. Idempotent per session."""
        chunks = chunk_resume(text)
        if not chunks:
            return []

        base_meta = dict(metadata or {})
        for chunk in chunks:
            chunk["metadata"].update(base_meta)

        stored = self.vector_store.add_resume(session_id, chunks)
        logger.info("Indexed resume for session %s: %d chunks", session_id, stored)
        return chunks[:stored] if stored else []

    def has_resume(self, session_id: str) -> bool:
        """True when the session has any indexed resume chunks."""
        return self.vector_store.count_session(session_id) > 0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def build_query(
        self,
        role: str,
        current_question: Optional[str] = None,
        last_answer: Optional[str] = None,
    ) -> str:
        """Build a focused retrieval query from the current interview context."""
        parts = [f"Role: {role or 'General'}"]
        if current_question:
            parts.append(f"Current interview question: {current_question}")
        if last_answer:
            parts.append(f"Candidate's last answer: {last_answer[:500]}")
        parts.append(
            "Find the candidate's most relevant resume details: project names, "
            "internships, certifications, skills, and measurable outcomes."
        )
        return "\n".join(parts)

    def retrieve(
        self,
        session_id: str,
        role: str,
        current_question: Optional[str] = None,
        last_answer: Optional[str] = None,
        top_k: int = _TOP_K_DEFAULT,
    ) -> List[Dict]:
        """Return ranked chunk dicts (text/metadata/distance) for a session.

        Drops chunks less similar than RAG_MAX_DISTANCE. Returns [] cleanly
        when the session has no resume.
        """
        if not session_id or not self.has_resume(session_id):
            return []

        query = self.build_query(role, current_question, last_answer)
        results = self.vector_store.query(session_id, query, k=top_k)
        relevant = [r for r in results if r.get("distance", 1.0) <= _MAX_DISTANCE]

        if _DEBUG_RAG:
            for r in relevant:
                logger.info(
                    "[RAG] session=%s section=%s dist=%.3f text=%.120s",
                    session_id,
                    r["metadata"].get("section"),
                    r.get("distance"),
                    r.get("text", "").replace("\n", " "),
                )
            dropped = len(results) - len(relevant)
            if dropped:
                logger.info("[RAG] dropped %d low-relevance chunk(s)", dropped)

        return relevant

    def retrieve_context(
        self,
        session_id: str,
        role: str,
        current_question: Optional[str] = None,
        last_answer: Optional[str] = None,
        top_k: int = _TOP_K_DEFAULT,
    ) -> Tuple[str, List[str]]:
        """Return (formatted context string, source section names).

        The context string is empty ("", []) when the session has no resume,
        so callers can branch cleanly.
        """
        results = self.retrieve(session_id, role, current_question, last_answer, top_k)
        if not results:
            return "", []

        # Group chunks by section for a readable, structured context block.
        by_section: Dict[str, List[str]] = {}
        for r in results:
            section = r["metadata"].get("section", "general")
            by_section.setdefault(section, []).append(r["text"].strip())

        sections = sorted(by_section)
        blocks = []
        for section in sections:
            lines = "\n".join(f"- {t}" for t in by_section[section])
            blocks.append(f"[{section}]\n{lines}")
        return "\n\n".join(blocks), sections
