"""Chunking utilities for resume text."""

from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split resume text into overlapping chunks suitable for retrieval."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk and chunk.strip()]
