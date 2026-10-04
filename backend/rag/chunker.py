"""Section-aware chunking for resume text.

Resumes have a predictable structure (headings like Experience, Education,
Projects, Skills). Splitting blindly on character counts can sever a project
description from its title, which hurts retrieval quality. This module first
segments the text by detected headings, then splits each section with a
RecursiveCharacterTextSplitter so chunks stay within one section.

Every chunk carries metadata: ``section`` (normalized heading name),
``chunk_index`` (global, deterministic), and ``section_chunk_index``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# (regex fragment, normalized section name). Matched against a whole line,
# case-insensitively, tolerating decorations like "EXPERIENCE:", "-- Skills --".
_SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"professional summary|summary|objective|profile", "summary"),
    (r"work experience|employment history|professional experience|experience", "experience"),
    (r"internships?|internship experience", "internships"),
    (r"key projects|selected projects|personal projects|projects", "projects"),
    (r"technical skills|core competencies|technologies|tech stack|skills", "skills"),
    (r"education|academic background|academic qualifications", "education"),
    (r"certifications|certificates|licenses", "certifications"),
    (r"achievements|awards|honors", "achievements"),
    (r"publications|research", "publications"),
    (r"languages", "languages"),
    (r"open source|open-source", "open_source"),
    (r"interests|hobbies", "interests"),
    (r"contact|personal details", "contact"),
]

# Compile whole-line matchers once. The decoration class strips common
# heading adornments (dashes, asterisks, colons) around the heading words.
_COMPILED_SECTIONS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"^[\s\-=–—*#_:.]*(" + fragment + r")[\s\-=–—*#_:.]*$",
            re.IGNORECASE,
        ),
        name,
    )
    for fragment, name in _SECTION_PATTERNS
]

# Lines longer than this are never treated as headings (avoids matching
# body sentences that happen to contain a keyword).
_MAX_HEADING_LINE_LENGTH = 60


def detect_section(line: str) -> str | None:
    """Return the normalized section name if the line is a resume heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LINE_LENGTH:
        return None
    # A heading is usually short and has few words.
    if len(stripped.split()) > 5:
        return None
    for pattern, name in _COMPILED_SECTIONS:
        if pattern.match(stripped):
            return name
    return None


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split resume text into (section_name, section_text) pairs.

    Text before the first detected heading is labelled ``"header"`` (this is
    usually the candidate's name and contact block).
    """
    sections: List[Tuple[str, List[str]]] = []
    current_name = "header"
    current_lines: List[str] = []

    for line in text.splitlines():
        section = detect_section(line)
        if section is not None:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = section
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    result = []
    for name, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            result.append((name, body))

    if not result and text.strip():
        result.append(("header", text.strip()))
    return result


def chunk_resume(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """Chunk resume text section-by-section.

    Returns a list of ``{"text": chunk, "metadata": {...}}`` dicts where
    metadata contains ``section``, ``chunk_index`` (global, deterministic)
    and ``section_chunk_index``.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict] = []
    chunk_index = 0
    for section_name, section_text in split_into_sections(text):
        # Prefix the section name so the embedding carries section context.
        pieces = splitter.split_text(section_text)
        for section_chunk_index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "text": piece,
                    "metadata": {
                        "section": section_name,
                        "chunk_index": chunk_index,
                        "section_chunk_index": section_chunk_index,
                    },
                }
            )
            chunk_index += 1

    logger.info(
        "Chunked resume into %d chunks across %d sections",
        len(chunks),
        len({c["metadata"]["section"] for c in chunks}),
    )
    return chunks


def split_text_into_chunks(
    text: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> List[str]:
    """Legacy plain-text splitter (kept for backwards compatibility)."""
    return [c["text"] for c in chunk_resume(text, chunk_size, chunk_overlap)]
