import os
import tempfile
import unittest
from pathlib import Path

from rag.extractor import extract_resume_text
from rag.retriever import ResumeRetriever
from rag.vector_store import ResumeVectorStore


def _fake_embed(texts):
    vectors = []
    for text in texts:
        vec = [0.0] * 16
        for char in text.lower():
            vec[ord(char) % 16] += 1.0
        norm = sum(value * value for value in vec) ** 0.5 or 1.0
        vectors.append([value / norm for value in vec])
    return vectors


def _minimal_pdf_bytes(text: str) -> bytes:
    """Build a minimal valid single-page PDF containing the given text."""
    body = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        b"4 0 obj << /Length " + str(len(body)).encode() + b" >> stream\n" + body + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    out, offsets = b"", []
    for i, part in enumerate(parts):
        if i > 0:  # skip the %PDF header; object offsets are 1-based
            offsets.append(len(out))
        out += part
    xref_at = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for pos in offsets:
        out += f"{pos:010d} 00000 n \n".encode()
    out += b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_at).encode() + b"\n%%EOF"
    return bytes(out)


class ResumeExtractionTests(unittest.TestCase):
    def test_extracts_text_from_docx(self):
        from docx import Document

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.docx"
            doc = Document()
            doc.add_heading("Software Engineer", level=1)
            doc.add_paragraph("Worked on Python APIs and React dashboards.")
            doc.add_paragraph("Built scalable services using Docker and Kubernetes.")
            doc.save(path)

            text = extract_resume_text(path.read_bytes(), path.name)

            self.assertIn("Software Engineer", text)
            self.assertIn("Python APIs", text)
            self.assertIn("Docker", text)

    def test_extracts_text_from_pdf(self):
        text = extract_resume_text(
            _minimal_pdf_bytes("Zebracorn billing platform"), "resume.pdf"
        )
        self.assertIn("Zebracorn", text)

    def test_rejects_scanned_pdf(self):
        with self.assertRaises(ValueError):
            extract_resume_text(_minimal_pdf_bytes("   "), "resume.pdf")


class ResumeIsolationTests(unittest.TestCase):
    def test_sessions_cannot_retrieve_each_others_resume(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResumeVectorStore(persist_directory=tmpdir)
            retriever = ResumeRetriever(vector_store=store)
            with patch("rag.vector_store.embed_texts", side_effect=_fake_embed), patch(
                "rag.retriever._MAX_DISTANCE", 10.0
            ):
                retriever.index_resume("session-a", "Alice built a Kubernetes billing platform in Go.")
                retriever.index_resume("session-b", "Bob taught kindergarten literacy workshops.")

                alice_context, _ = retriever.retrieve_context(
                    "session-a", role="Software Engineer", top_k=3
                )
                bob_context, _ = retriever.retrieve_context(
                    "session-b", role="Teacher", top_k=3
                )

            self.assertTrue(alice_context)
            self.assertTrue(bob_context)
            self.assertIn("Kubernetes", alice_context)
            self.assertNotIn("kindergarten", alice_context.lower())
            self.assertNotIn("kubernetes", bob_context.lower())
            store.close()


if __name__ == "__main__":
    unittest.main()
