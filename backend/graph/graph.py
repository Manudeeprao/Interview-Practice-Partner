"""
LangGraph StateGraph definition for the Interview Practice Partner.

This module assembles all nodes into an explicit state graph with
conditional edges for routing based on classify_answer_node output.

Graph topology:
  START
    └─► role_intake_node  ─────────────────────────────────────────────────┐
          │ (role confirmed)                                                 │
          ▼                                                                 │ (role not confirmed)
    ask_question_node  ◄──────────────────────────────────────────┐         │
          │                                                        │         │
          ▼ (awaits user input — handled by Flask per-turn)        │         │
    classify_answer_node                                           │         │
          │                                                        │         │
          ├─ VAGUE ──────► follow_up_node ──────────────────────►─┘         │
          ├─ GOOD ───────► next_question_node ──(enough?)──► generate_feedback_node → END
          │                       │ (not done)                              │
          │                       └─────────────────────────────────────────┘
          ├─ OFF_TOPIC ──► redirect_node ───────────────────────►─┘
          └─ OUT_OF_SCOPE► decline_node ────────────────────────►─┘

Note: Because this is a voice/chat app that pauses for user input between
turns, the graph is compiled but invoked one-step-at-a-time by Flask.
The `next_node` field in state tells Flask which node to run next turn.
"""

import logging
from langgraph.graph import StateGraph, END

from .state import InterviewState
from .nodes import (
    role_intake_node,
    ask_question_node,
    classify_answer_node,
    follow_up_node,
    next_question_node,
    redirect_node,
    decline_node,
    generate_feedback_node,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------
def route_after_role_intake(state: InterviewState) -> str:
    """After role intake, either loop (clarify role) or proceed to ask questions."""
    next_node = state.get("next_node", "role_intake_node")
    if state.get("role_confirmed"):
        return "ask_question_node"
    return "role_intake_node"


def route_after_classify(state: InterviewState) -> str:
    """Routes based on classification result from classify_answer_node."""
    classification = state.get("classification", "GOOD")
    routing_map = {
        "VAGUE": "follow_up_node",
        "GOOD": "next_question_node",
        "OFF_TOPIC": "redirect_node",
        "OUT_OF_SCOPE": "decline_node",
    }
    destination = routing_map.get(classification, "next_question_node")
    logger.debug("route_after_classify: %s → %s", classification, destination)
    return destination


def route_after_next_question(state: InterviewState) -> str:
    """Decides whether to continue interviewing or generate feedback."""
    if state.get("done") or state.get("next_node") == "generate_feedback_node":
        return "generate_feedback_node"
    return "ask_question_node"


def route_after_follow_up(state: InterviewState) -> str:
    """Follow-up always leads back to classify_answer_node for the next answer."""
    return "classify_answer_node"


def route_after_redirect(state: InterviewState) -> str:
    """Redirect always leads back to classify_answer_node."""
    return "classify_answer_node"


def route_after_decline(state: InterviewState) -> str:
    """Decline always leads back to classify_answer_node."""
    return "classify_answer_node"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_interview_graph():
    """
    Constructs and compiles the LangGraph StateGraph.
    
    Returns the compiled graph object, ready to be invoked per turn.
    """
    graph = StateGraph(InterviewState)

    # Register all nodes
    graph.add_node("role_intake_node", role_intake_node)
    graph.add_node("ask_question_node", ask_question_node)
    graph.add_node("classify_answer_node", classify_answer_node)
    graph.add_node("follow_up_node", follow_up_node)
    graph.add_node("next_question_node", next_question_node)
    graph.add_node("redirect_node", redirect_node)
    graph.add_node("decline_node", decline_node)
    graph.add_node("generate_feedback_node", generate_feedback_node)

    # Entry point
    graph.set_entry_point("role_intake_node")

    # Conditional edge from role_intake_node
    graph.add_conditional_edges(
        "role_intake_node",
        route_after_role_intake,
        {
            "role_intake_node": "role_intake_node",
            "ask_question_node": "ask_question_node",
        },
    )

    # ask_question_node → awaits user (graph pauses here between turns)
    # When user replies, Flask re-enters at classify_answer_node
    graph.add_edge("ask_question_node", END)

    # Conditional edges from classify_answer_node
    graph.add_conditional_edges(
        "classify_answer_node",
        route_after_classify,
        {
            "follow_up_node": "follow_up_node",
            "next_question_node": "next_question_node",
            "redirect_node": "redirect_node",
            "decline_node": "decline_node",
        },
    )

    # Conditional edge from next_question_node
    graph.add_conditional_edges(
        "next_question_node",
        route_after_next_question,
        {
            "ask_question_node": "ask_question_node",
            "generate_feedback_node": "generate_feedback_node",
        },
    )

    # follow_up_node, redirect_node, decline_node all end their sub-turn
    # (The agent response is returned; next user input re-enters at classify_answer_node)
    graph.add_edge("follow_up_node", END)
    graph.add_edge("redirect_node", END)
    graph.add_edge("decline_node", END)

    # generate_feedback_node → END
    graph.add_edge("generate_feedback_node", END)

    compiled = graph.compile()
    logger.info("LangGraph interview graph compiled successfully.")
    return compiled


# NOTE: The compiled graph object is available via build_interview_graph() for
# introspection or testing. Flask's run_turn() in app.py dispatches nodes
# directly (one node per HTTP turn) rather than running the full graph at once,
# because the interview pauses between turns to wait for user input.
# Call build_interview_graph() explicitly if you want the compiled graph object.
