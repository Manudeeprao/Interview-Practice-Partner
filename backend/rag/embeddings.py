"""Embedding utilities for the resume RAG pipeline.

The sentence-transformers model is loaded lazily exactly once per process
(module-level singleton) and reused for every call. Override the model with
the EMBEDDING_MODEL environment variable if needed.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers model once and reuse it."""
    global _model
    if _model is None:
        logger.info("Loading embedding model %s", _EMBEDDING_MODEL)
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def reset_embedding_model() -> None:
    """Test helper: drop the cached model so tests can reconfigure it."""
    global _model
    _model = None


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Create embeddings for a batch of text chunks (batched encode)."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=False,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [list(map(float, embedding)) for embedding in embeddings]
