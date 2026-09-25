from graph.graph import route_after_classify
from graph.nodes import _classification_to_node, classify_answer_node
from graph.state import create_initial_state


def _active_state(message: str) -> dict:
    state = create_initial_state("route-test")
    state.update({
        "role": "Software Engineer",
        "role_confirmed": True,
        "current_question": "Tell me about a challenging project.",
        "last_user_message": message,
        "interview_stage": "project_discussion",
    })
    return state


def test_classification_maps_to_expected_nodes():
    assert _classification_to_node("GOOD") == "next_question_node"
    assert _classification_to_node("VAGUE") == "follow_up_node"
    assert _classification_to_node("OFF_TOPIC") == "redirect_node"
    assert _classification_to_node("OUT_OF_SCOPE") == "decline_node"


def test_graph_router_matches_classifier_labels():
    for label, node in {
        "GOOD": "next_question_node",
        "VAGUE": "follow_up_node",
        "OFF_TOPIC": "redirect_node",
        "OUT_OF_SCOPE": "decline_node",
    }.items():
        assert route_after_classify({"classification": label}) == node


def test_classify_good_route(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.structured_completion",
        lambda **kwargs: {"classification": "GOOD", "strength": "strong", "reason": "detailed"},
    )
    result = classify_answer_node(_active_state("I designed a REST API with retries and reduced p95 latency by 40%."))
    assert result["classification"] == "GOOD"
    assert result["next_node"] == "next_question_node"


def test_classify_vague_route(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.structured_completion",
        lambda **kwargs: {"classification": "VAGUE", "strength": "weak", "reason": "brief"},
    )
    result = classify_answer_node(_active_state("I did stuff."))
    assert result["classification"] == "VAGUE"
    assert result["next_node"] == "follow_up_node"


def test_classify_off_topic_route(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.structured_completion",
        lambda **kwargs: {"classification": "OFF_TOPIC", "strength": "weak", "reason": "tangent"},
    )
    result = classify_answer_node(_active_state("Let's talk about pepperoni pizza."))
    assert result["classification"] == "OFF_TOPIC"
    assert result["next_node"] == "redirect_node"
    assert result["is_follow_up"] is False


def test_classify_out_of_scope_route(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.structured_completion",
        lambda **kwargs: {"classification": "OUT_OF_SCOPE", "strength": "weak", "reason": "request"},
    )
    result = classify_answer_node(_active_state("Write my resume for me."))
    assert result["classification"] == "OUT_OF_SCOPE"
    assert result["next_node"] == "decline_node"
