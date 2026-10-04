"""Tests for the resume RAG pipeline: chunking, isolation, re-upload, retrieval."""

import hashlib
import math

import pytest

from rag.chunker import chunk_resume, detect_section, split_into_sections
from rag.retriever import ResumeRetriever
from rag.vector_store import ResumeVectorStore


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (avoids downloading the transformer model)
# ---------------------------------------------------------------------------
def _fake_embed(texts, batch_size=32):
    vectors = []
    for text in texts:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        vec = [b / 255.0 for b in digest[:16]]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr("rag.vector_store.embed_texts", _fake_embed)
    # Accept every retrieved chunk regardless of distance in plumbing tests.
    monkeypatch.setattr("rag.retriever._MAX_DISTANCE", 10.0)


@pytest.fixture
def store(tmp_path):
    return ResumeVectorStore(persist_directory=str(tmp_path / "chroma"))


@pytest.fixture
def retriever(store):
    return ResumeRetriever(vector_store=store)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def test_detect_section_headings():
    assert detect_section("Experience") == "experience"
    assert detect_section("WORK EXPERIENCE:") == "experience"
    assert detect_section("-- Technical Skills --") == "skills"
    assert detect_section("Education") == "education"
    assert detect_section("Projects") == "projects"
    # Body sentences must not match.
    assert detect_section("I have five years of experience building web apps.") is None
    assert detect_section("") is None


def test_split_into_sections_labels_header():
    text = "Jane Doe\njane@example.com\n\nExperience\nBuilt things.\n\nSkills\nPython"
    sections = dict(split_into_sections(text))
    assert sections["header"].startswith("Jane Doe")
    assert "Built things." in sections["experience"]
    assert "Python" in sections["skills"]


def test_chunk_resume_metadata():
    text = (
        "Jane Doe\n\nExperience\nLed a team building a billing API. " * 4
        + "\n\nSkills\nPython, Go, SQL. " * 4
    )
    chunks = chunk_resume(text, chunk_size=120, chunk_overlap=20)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        meta = chunk["metadata"]
        assert "section" in meta and "chunk_index" in meta
        assert chunk["text"].strip()
    indices = [c["metadata"]["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index must be deterministic"
    sections = {c["metadata"]["section"] for c in chunks}
    assert {"experience", "skills"} <= sections


def test_chunk_resume_empty():
    assert chunk_resume("") == []
    assert chunk_resume("   ") == []


# ---------------------------------------------------------------------------
# Indexing / isolation / re-upload
# ---------------------------------------------------------------------------
def test_per_session_isolation(retriever, fake_embeddings):
    retriever.index_resume("session-a", "Experience\nBuilt the Zebracorn payments platform.")
    retriever.index_resume("session-b", "Experience\nMaintained the Quux inventory system.")

    context_a, _ = retriever.retrieve_context("session-a", role="Software Engineer")
    context_b, _ = retriever.retrieve_context("session-b", role="Software Engineer")

    assert "Zebracorn" in context_a
    assert "Quux" not in context_a
    assert "Quux" in context_b
    assert "Zebracorn" not in context_b


def test_reupload_replaces_old_chunks(retriever, store, fake_embeddings):
    retriever.index_resume("s1", "Experience\nBuilt the Zebracorn platform with Python.")
    first_count = store.count_session("s1")
    assert first_count > 0

    retriever.index_resume("s1", "Skills\nRust and WebAssembly.")
    assert store.count_session("s1") > 0

    context, _ = retriever.retrieve_context("s1", role="Software Engineer")
    assert "Zebracorn" not in context
    assert "Rust" in context


def test_retrieve_empty_without_resume(retriever, fake_embeddings):
    context, sections = retriever.retrieve_context("no-such-session", role="Data Analyst")
    assert context == ""
    assert sections == []


def test_delete_session_clears_vectors(retriever, store, fake_embeddings):
    retriever.index_resume("s9", "Experience\nDid things.")
    assert store.count_session("s9") > 0
    store.delete_session("s9")
    assert store.count_session("s9") == 0


def test_context_format_groups_by_section(retriever, fake_embeddings):
    text = "Jane Doe\n\nExperience\nLed the Zebracorn project.\n\nSkills\nPython, SQL."
    retriever.index_resume("fmt", text)
    context, sections = retriever.retrieve_context("fmt", role="Software Engineer")
    assert "[experience]" in context
    assert "[skills]" in context
    assert set(sections) >= {"experience", "skills"}


# ---------------------------------------------------------------------------
# Query construction + relevance threshold
# ---------------------------------------------------------------------------
def test_build_query_uses_current_context_not_history():
    retriever = ResumeRetriever.__new__(ResumeRetriever)  # no vector store needed
    query = ResumeRetriever.build_query(
        retriever,
        role="Data Scientist",
        current_question="Explain cross-validation.",
        last_answer="I used k-fold on the churn model.",
    )
    assert "Data Scientist" in query
    assert "cross-validation" in query
    assert "churn model" in query


def test_threshold_drops_irrelevant(retriever, monkeypatch):
    monkeypatch.setattr(
        "rag.retriever.ResumeVectorStore.query",
        lambda self, session_id, query_text, k=5: [
            {"text": "close chunk", "metadata": {"section": "skills"}, "distance": 0.1},
            {"text": "far chunk", "metadata": {"section": "skills"}, "distance": 0.95},
        ],
    )
    monkeypatch.setattr("rag.retriever.ResumeRetriever.has_resume", lambda self, sid: True)
    monkeypatch.setattr("rag.retriever._MAX_DISTANCE", 0.5)

    results = retriever.retrieve("s1", role="Software Engineer")
    assert [r["text"] for r in results] == ["close chunk"]
