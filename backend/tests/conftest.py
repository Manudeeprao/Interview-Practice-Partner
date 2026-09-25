import os
from pathlib import Path

os.environ.setdefault("IPP_TESTING", "1")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

import pytest
from langgraph.checkpoint.memory import MemorySaver

from graph.graph import build_interview_graph
from rag.retriever import ResumeRetriever
from rag.vector_store import ResumeVectorStore
from storage import SessionStore


def make_fake_structured():
    def fake_structured(system_prompt, user_content, **kwargs):
        prompt = (system_prompt or "").lower()
        content = (user_content or "").lower()
        if "quality controller" in prompt:
            return {"aligned": True, "reason": "aligned with the selected role"}
        if "interview coach" in prompt:
            return {
                "readinessScore": 7,
                "overallImpression": "Solid session.",
                "communication": "Clear and structured.",
                "technicalKnowledge": "Demonstrated relevant skills.",
                "strengths": "Used concrete examples.",
                "improvementAreas": ["Add more metrics to outcomes."],
            }
        if "write my resume" in content or "give me the answer" in content:
            return {"classification": "OUT_OF_SCOPE", "strength": "weak", "reason": "out of scope request"}
        if "pepperoni" in content or "weather" in content:
            return {"classification": "OFF_TOPIC", "strength": "weak", "reason": "unrelated tangent"}
        if "i did stuff" in content or "not sure" in content:
            return {"classification": "VAGUE", "strength": "weak", "reason": "too brief"}
        return {"classification": "GOOD", "strength": "strong", "reason": "detailed and relevant"}

    return fake_structured


def make_fake_chat():
    def fake_chat(system_prompt, history, user_message, **kwargs):
        return "What is one technically interesting decision you made recently?"

    return fake_chat


@pytest.fixture
def memory_graph():
    return build_interview_graph(checkpointer=MemorySaver())


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(db_path=str(tmp_path / "sessions.sqlite"), max_sessions=20)


@pytest.fixture
def llm_mocks(monkeypatch):
    monkeypatch.setattr("graph.nodes.chat_completion", make_fake_chat())
    monkeypatch.setattr("graph.nodes.structured_completion", make_fake_structured())


@pytest.fixture
def client(tmp_path, memory_graph, session_store, llm_mocks):
    from app import create_app

    chroma_dir = tmp_path / "chroma"
    retriever = ResumeRetriever(vector_store=ResumeVectorStore(persist_directory=str(chroma_dir)))
    flask_app = create_app(
        session_store=session_store,
        interview_graph=memory_graph,
        resume_retriever=retriever,
    )
    flask_app.config["TESTING"] = True
    return flask_app.test_client()
