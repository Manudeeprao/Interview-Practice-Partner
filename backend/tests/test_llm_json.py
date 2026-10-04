"""Tests for robust structured LLM output parsing."""

import pytest

from graph.llm import _extract_json, structured_completion


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_multiline():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_fenced_single_line():
    assert _extract_json('```{"a": 1}```') == {"a": 1}


def test_extract_json_with_prose():
    assert _extract_json('Here is the result:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_extract_json_rejects_non_objects():
    assert _extract_json("[1, 2, 3]") is None
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None
    assert _extract_json(None) is None


def test_structured_completion_retries_once_on_bad_json(monkeypatch):
    calls = []

    def fake_call(messages, temperature=0.1, max_tokens=256):
        calls.append(messages)
        if len(calls) == 1:
            return "this is not json"
        return '{"classification": "GOOD", "strength": "adequate", "reason": "ok"}'

    monkeypatch.setattr("graph.llm._call_groq", fake_call)
    result = structured_completion("system", "classify this")
    assert result == {"classification": "GOOD", "strength": "adequate", "reason": "ok"}
    assert len(calls) == 2


def test_structured_completion_gives_up_after_retry(monkeypatch):
    monkeypatch.setattr("graph.llm._call_groq", lambda **kwargs: "still not json")
    assert structured_completion("system", "classify this") is None


def test_feedback_fallback_when_llm_down(monkeypatch):
    from graph.nodes import generate_feedback_node
    from graph.state import create_initial_state

    monkeypatch.setattr("graph.nodes.structured_completion", lambda **kwargs: None)
    state = create_initial_state("fb-fallback")
    state.update(
        {
            "role": "Software Engineer",
            "history": [{"role": "user", "content": "I built APIs."}],
            "main_question_count": 3,
        }
    )
    result = generate_feedback_node(state)
    assert result["done"] is True
    feedback = result["feedback"]
    assert feedback["readinessScore"] is None
    assert "Software Engineer" in feedback["overallImpression"]
    assert isinstance(feedback["improvementAreas"], list)
