"""Embedding utilities for the resume RAG pipeline."""

from __future__ import annotations

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers model once and reuse it."""
    global _model
    if _model is None:
        logger.info("Loading embedding model %s", _EMBEDDING_MODEL)
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a batch of text chunks."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
    return [list(map(float, embedding)) for embedding in embeddings]
