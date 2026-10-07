"""Tests for the resume-QA path: direct questions about the candidate's own
resume must be answered from the indexed resume via RAG, not misclassified
as OFF_TOPIC / OUT_OF_SCOPE."""

import pytest

from graph.graph import build_interview_graph, route_after_classify
from graph.nodes import (
    _is_resume_question,
    classify_answer_node,
    resume_qa_node,
)
from graph.state import create_initial_state


# ---------------------------------------------------------------------------
# _is_resume_question heuristic
# ---------------------------------------------------------------------------

TRUE_CASES = [
    "what is my name?",
    "what is my name , i mean see my details in my resume",
    "ask from my resume",
    "tell me about my projects",
    "list my skills",
    "what certifications do i have?",
    "show me my experience",
    "summarize my resume",
    "do i have any internships?",
    "can you read my cv?",
]

FALSE_CASES = [
    # ordinary interview answers mentioning "my project" must not match
    "I built my project with React and deployed it on Vercel",
    "My internship at TCS taught me teamwork",
    "for software engineer role",
    "what is a REST API?",
    "I have 3 years of experience with Python",
    "stop",  # stop signal shape, no resume cue
    "",
]


@pytest.mark.parametrize("text", TRUE_CASES)
def test_is_resume_question_true(text):
    assert _is_resume_question(text) is True


@pytest.mark.parametrize("text", FALSE_CASES)
def test_is_resume_question_false(text):
    assert _is_resume_question(text) is False


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_after_classify_resume_qa():
    assert route_after_classify({"classification": "RESUME_QA"}) == "resume_qa_node"


def test_graph_has_resume_qa_node():
    graph = build_interview_graph()
    assert "resume_qa_node" in graph.nodes


# ---------------------------------------------------------------------------
# classify_answer_node routes resume questions without an LLM call
# ---------------------------------------------------------------------------


def _state_with_resume(message):
    state = create_initial_state("sess-qa-1")
    state.update(
        {
            "role": "Software Engineer",
            "role_confirmed": True,
            "resume_uploaded": True,
            "resume_filename": "resume.pdf",
            "resume_chunk_count": 8,
            "current_question": "Tell me about yourself.",
            "last_user_message": message,
        }
    )
    return state


def test_classify_routes_resume_question_to_resume_qa(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM classifier must not be called for resume questions")

    monkeypatch.setattr("graph.nodes.structured_completion", fail_if_called)
    result = classify_answer_node(_state_with_resume("what is my name?"))
    assert result["classification"] == "RESUME_QA"
    assert result["next_node"] == "resume_qa_node"
    # user turn still recorded in history
    assert result["history"][-1] == {"role": "user", "content": "what is my name?"}


def test_classify_ignores_resume_question_without_upload(monkeypatch):
    calls = []

    def fake_structured(system_prompt, user_content, **kwargs):
        calls.append(user_content)
        return {"classification": "GOOD", "strength": "adequate", "reason": "ok"}

    monkeypatch.setattr("graph.nodes.structured_completion", fake_structured)
    state = _state_with_resume("what is my name?")
    state["resume_uploaded"] = False
    result = classify_answer_node(state)
    assert result["classification"] == "GOOD"
    assert calls, "LLM classifier should handle it when no resume is uploaded"


# ---------------------------------------------------------------------------
# resume_qa_node
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self, context="(no context)", sections=None):
        self._context = context
        self._sections = sections or []
        self.seen_queries = []

    def retrieve_context(self, session_id, role, **kwargs):
        self.seen_queries.append(kwargs.get("direct_query"))
        return self._context, self._sections


def _patch_qa(monkeypatch, retriever, answer):
    monkeypatch.setattr("graph.nodes.get_resume_retriever", lambda: retriever)
    monkeypatch.setattr(
        "graph.nodes.chat_completion",
        lambda system_prompt, history, user_message, **kwargs: answer,
    )


def test_resume_qa_node_answers_from_context(monkeypatch):
    retriever = _FakeRetriever(
        context="[header]\n- John Doe\n- john@example.com",
        sections=["header"],
    )
    _patch_qa(monkeypatch, retriever, "Your name is John Doe.")
    result = resume_qa_node(_state_with_resume("what is my name?"))
    assert result["agent_response"] == "Your name is John Doe."
    assert result["history"][-1] == {"role": "assistant", "content": "Your name is John Doe."}
    assert result["next_node"] == "classify_answer_node"
    # the verbatim user question must be the retrieval query (no boilerplate)
    assert retriever.seen_queries == ["what is my name?"]


def test_resume_qa_node_honest_when_no_context(monkeypatch):
    retriever = _FakeRetriever(context="", sections=[])
    captured = {}

    def fake_chat(system_prompt, history, user_message, **kwargs):
        captured["instruction"] = user_message
        return "I don't have that detail."

    monkeypatch.setattr("graph.nodes.get_resume_retriever", lambda: retriever)
    monkeypatch.setattr("graph.nodes.chat_completion", fake_chat)
    result = resume_qa_node(_state_with_resume("what is my name?"))
    assert result["agent_response"] == "I don't have that detail."
    assert "No resume context was retrieved" in captured["instruction"]


def test_resume_qa_node_rate_limit_fallback(monkeypatch):
    retriever = _FakeRetriever(context="[header]\n- Jane", sections=["header"])
    _patch_qa(monkeypatch, retriever, None)  # chat_completion returns None
    result = resume_qa_node(_state_with_resume("what is my name?"))
    assert "rate limit" in result["agent_response"].lower()
    assert "30 seconds" in result["agent_response"]


def test_resume_qa_node_does_not_advance_question_count(monkeypatch):
    retriever = _FakeRetriever(context="[header]\n- Jane", sections=["header"])
    _patch_qa(monkeypatch, retriever, "Jane.")
    state = _state_with_resume("what is my name?")
    state["main_question_count"] = 2
    result = resume_qa_node(state)
    assert "main_question_count" not in result
    assert "difficulty" not in result
