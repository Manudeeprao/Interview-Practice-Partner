from graph.graph import build_interview_graph, route_after_classify, route_after_next_question
from langgraph.checkpoint.memory import MemorySaver
from graph.state import create_initial_state


def test_graph_compiles_with_expected_nodes():
    compiled = build_interview_graph(checkpointer=MemorySaver())
    graph = compiled.get_graph()
    node_ids = set(graph.nodes)
    for expected in {
        "role_intake_node",
        "ask_question_node",
        "classify_answer_node",
        "follow_up_node",
        "next_question_node",
        "redirect_node",
        "decline_node",
        "generate_feedback_node",
    }:
        assert expected in node_ids


def test_compiled_graph_executes_role_intake_and_first_question(llm_mocks):
    compiled = build_interview_graph(checkpointer=MemorySaver())
    state = create_initial_state("graph-exec-1")
    state["last_user_message"] = "I want a software engineer interview"
    result = compiled.invoke(state, config={"configurable": {"thread_id": "graph-exec-1"}})
    assert result["role_confirmed"] is True
    assert result["role"] == "Software Engineer"
    assert result["current_question"]
    assert result["next_node"] == "classify_answer_node"


def test_route_helpers_do_not_use_flask_dispatch():
    assert route_after_classify({"classification": "VAGUE"}) == "follow_up_node"
    assert route_after_next_question({"ready_for_feedback": True}) == "generate_feedback_node"
    assert route_after_next_question({"done": False, "ready_for_feedback": False}) == "ask_question_node"
