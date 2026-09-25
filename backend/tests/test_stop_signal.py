from graph.nodes import _contains_stop_signal, classify_answer_node, next_question_node
from graph.state import create_initial_state


def test_stop_phrases_are_detected():
    assert _contains_stop_signal("end the interview")
    assert _contains_stop_signal("I'm done with the interview")
    assert _contains_stop_signal("generate my feedback")


def test_natural_language_done_is_not_a_stop():
    assert not _contains_stop_signal("I was done with the migration after we finished testing.")
    assert not _contains_stop_signal("We had to stop the job because of a bad deploy.")


def test_classifier_routes_stop_to_next_question():
    state = create_initial_state("stop-1")
    state.update({
        "role_confirmed": True,
        "current_question": "Tell me about yourself.",
        "last_user_message": "end the interview",
    })
    result = classify_answer_node(state)
    assert result["next_node"] == "next_question_node"
    assert "end" in result["classification_reason"].lower() or "requested" in result["classification_reason"].lower()


def test_early_stop_is_rejected_before_three_questions():
    state = create_initial_state("stop-2")
    state.update({
        "last_user_message": "end the interview",
        "main_question_count": 0,
        "max_questions": 7,
        "history": [],
    })
    result = next_question_node(state)
    assert result["ready_for_feedback"] is False
    assert result["next_node"] == "ask_question_node"
    assert result["main_question_count"] == 1


def test_stop_after_minimum_questions_goes_to_feedback():
    state = create_initial_state("stop-3")
    state.update({
        "last_user_message": "generate my feedback",
        "main_question_count": 3,
        "max_questions": 7,
        "history": [],
    })
    result = next_question_node(state)
    assert result["ready_for_feedback"] is True
    assert result["next_node"] == "generate_feedback_node"
