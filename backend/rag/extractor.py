"""Resume text extraction for PDF and DOCX files."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _xpath(element, expr, namespaces):
    try:
        return element.xpath(expr, namespaces=namespaces)
    except TypeError:
        return element.xpath(expr, namespaces)


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a PDF or DOCX resume.

    The implementation preserves heading-like structure where possible by keeping
    paragraph breaks and by preserving document headings from DOCX.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read PDF resume: %s", exc)
            raise ValueError("Unable to read the uploaded PDF resume.") from exc

        pieces: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pieces.append(page_text)
        text = "\n\n".join(pieces)

        if not text.strip():
            raise ValueError(
                "The uploaded PDF resume did not contain any readable text. "
                "If this is a scanned or image-only PDF, please upload a text-based PDF or DOCX file."
            )

    elif suffix == ".docx":
        try:
            doc = Document(io.BytesIO(file_bytes))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read DOCX resume: %s", exc)
            raise ValueError("Unable to read the uploaded DOCX resume.") from exc

        paragraphs = []
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        try:
            for paragraph in _xpath(doc.element.body, './/w:p', namespace):
                text_nodes = _xpath(paragraph, './/w:t', namespace)
                paragraph_text = ''.join([node.text for node in text_nodes if node.text])
                paragraph_text = paragraph_text.strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
        except Exception:
            paragraphs = []

        text = "\n".join(paragraphs)

        if not text:
            # Fallback to paragraph iteration for compatibility with older python-docx versions.
            paragraphs = []
            for paragraph in doc.paragraphs:
                paragraph_text = paragraph.text.strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
            text = "\n".join(paragraphs)

    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX resume.")

    normalized = _normalize_whitespace(text)
    if not normalized:
        raise ValueError("The uploaded resume did not contain any readable text.")
    return normalized
