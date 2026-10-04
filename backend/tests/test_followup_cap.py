"""Tests for the follow-up cap: the graph must never loop on VAGUE answers."""

from graph.nodes import MAX_FOLLOW_UPS_PER_QUESTION, classify_answer_node
from graph.state import create_initial_state


def _active_state(message: str, follow_ups_this_question: int = 0) -> dict:
    state = create_initial_state("cap-test")
    state.update(
        {
            "role": "Software Engineer",
            "role_confirmed": True,
            "current_question": "Tell me about a challenging project.",
            "last_user_message": message,
            "interview_stage": "project_discussion",
            "follow_ups_this_question": follow_ups_this_question,
        }
    )
    return state


def _vague_classifier(**kwargs):
    return {"classification": "VAGUE", "strength": "weak", "reason": "too brief"}


def test_below_cap_still_follows_up(monkeypatch):
    monkeypatch.setattr("graph.nodes.structured_completion", _vague_classifier)
    result = classify_answer_node(_active_state("I did stuff.", follow_ups_this_question=1))
    assert result["classification"] == "VAGUE"
    assert result["next_node"] == "follow_up_node"


def test_cap_reached_advances_interview(monkeypatch):
    """After MAX_FOLLOW_UPS_PER_QUESTION vague answers, classify as GOOD so the
    conditional-edge router sends the turn to next_question_node."""
    monkeypatch.setattr("graph.nodes.structured_completion", _vague_classifier)
    result = classify_answer_node(
        _active_state("I did stuff.", follow_ups_this_question=MAX_FOLLOW_UPS_PER_QUESTION)
    )
    assert result["classification"] == "GOOD"
    assert result["next_node"] == "next_question_node"
    assert "cap" in result["classification_reason"].lower()


def test_fail_open_path_respects_cap(monkeypatch):
    """When the classifier itself is down and the cap is reached, the graph
    must advance instead of looping on follow-ups forever."""
    monkeypatch.setattr("graph.nodes.structured_completion", lambda **kwargs: None)
    result = classify_answer_node(
        _active_state("anything", follow_ups_this_question=MAX_FOLLOW_UPS_PER_QUESTION)
    )
    assert result["next_node"] == "next_question_node"


def test_fail_open_path_follows_up_below_cap(monkeypatch):
    monkeypatch.setattr("graph.nodes.structured_completion", lambda **kwargs: None)
    result = classify_answer_node(_active_state("anything", follow_ups_this_question=0))
    assert result["next_node"] == "follow_up_node"


def test_cap_value_is_two():
    assert MAX_FOLLOW_UPS_PER_QUESTION == 2
