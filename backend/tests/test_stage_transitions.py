from graph.nodes import _compute_stage, next_question_node
from graph.state import create_initial_state


def test_stage_mapping_for_seven_question_interview():
    assert _compute_stage(0, 7) == "introduction"
    assert _compute_stage(1, 7) == "project_discussion"
    assert _compute_stage(2, 7) == "project_discussion"
    assert _compute_stage(3, 7) == "technical_fundamentals"
    assert _compute_stage(4, 7) == "technical_fundamentals"
    assert _compute_stage(5, 7) == "system_design"
    assert _compute_stage(6, 7) == "behavioral"


def test_good_answer_advances_stage():
    state = create_initial_state("stage-1")
    state.update({
        "last_user_message": "I led the API rewrite and cut p95 latency by 40%.",
        "main_question_count": 0,
        "max_questions": 7,
        "history": [],
        "is_follow_up": False,
    })
    result = next_question_node(state)
    assert result["main_question_count"] == 1
    assert result["interview_stage"] == "project_discussion"
    assert result["next_node"] == "ask_question_node"


def test_completing_max_questions_moves_to_feedback_stage():
    state = create_initial_state("stage-2")
    state.update({
        "last_user_message": "In that situation I facilitated a retro and reset the timeline.",
        "main_question_count": 6,
        "max_questions": 7,
        "history": [],
    })
    result = next_question_node(state)
    assert result["main_question_count"] == 7
    assert result["ready_for_feedback"] is True
    assert result["interview_stage"] == "feedback"
