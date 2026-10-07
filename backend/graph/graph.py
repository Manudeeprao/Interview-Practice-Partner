"""
Compiled LangGraph StateGraph for the Interview Practice Partner.

Routing is done by conditional edges — not by Flask reading next_node.
The compiled graph pauses with interrupt_before before nodes that need
the next user message, then Flask resumes the same checkpoint.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    ask_question_node,
    classify_answer_node,
    decline_node,
    follow_up_node,
    generate_feedback_node,
    next_question_node,
    redirect_node,
    resume_qa_node,
    role_intake_node,
)
from .state import InterviewState

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "checkpoints.sqlite",
)

_compiled_graph = None
_checkpointer_conn = None


def route_after_role_intake(state: InterviewState) -> str:
    """After role intake, either wait for clarification or ask the first question."""
    if state.get("role_confirmed"):
        return "ask_question_node"
    return "wait_for_role_input_node"


def route_at_start(state: InterviewState) -> str:
    """Entry routing for a new invoke when no interrupt is being resumed."""
    if state.get("done"):
        return "generate_feedback_node"
    if state.get("role_confirmed"):
        return "classify_answer_node"
    return "role_intake_node"


def route_after_classify(state: InterviewState) -> str:
    """Routes based on classification result from classify_answer_node."""
    classification = state.get("classification", "GOOD")
    routing_map = {
        "VAGUE": "follow_up_node",
        "GOOD": "next_question_node",
        "OFF_TOPIC": "redirect_node",
        "OUT_OF_SCOPE": "decline_node",
        "RESUME_QA": "resume_qa_node",
    }
    destination = routing_map.get(classification, "next_question_node")
    logger.debug("route_after_classify: %s → %s", classification, destination)
    return destination


def route_after_next_question(state: InterviewState) -> str:
    """Decides whether to continue interviewing or generate feedback."""
    if state.get("done") or state.get("ready_for_feedback"):
        return "generate_feedback_node"
    return "ask_question_node"


def route_after_ask_question(state: InterviewState) -> str:
    """Retry question generation when the LLM could not produce a question."""
    if state.get("question_generation_failed"):
        return "wait_for_question_retry_node"
    return "classify_answer_node"


def wait_for_role_input_node(state: InterviewState) -> dict:
    """Checkpoint-only node used while waiting for role clarification."""
    return {"next_node": "role_intake_node"}


def wait_for_question_retry_node(state: InterviewState) -> dict:
    """Checkpoint-only node used before retrying failed question generation."""
    return {"question_generation_failed": False, "next_node": "ask_question_node"}


def _make_sqlite_checkpointer(db_path: str):
    """Create a durable LangGraph checkpointer backed by SQLite."""
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        logger.warning(
            "langgraph.checkpoint.sqlite is unavailable (%s); falling back to MemorySaver.",
            exc,
        )
        return MemorySaver(), None

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
    except TypeError:
        saver = SqliteSaver.from_conn_string(db_path)
        conn.close()
        conn = None
    return saver, conn


def build_interview_graph(checkpointer=None):
    """
    Constructs and compiles the LangGraph StateGraph.

    Returns the compiled graph object. Flask resumes it per HTTP turn.
    """
    graph = StateGraph(InterviewState)

    graph.add_node("role_intake_node", role_intake_node)
    graph.add_node("ask_question_node", ask_question_node)
    graph.add_node("classify_answer_node", classify_answer_node)
    graph.add_node("follow_up_node", follow_up_node)
    graph.add_node("next_question_node", next_question_node)
    graph.add_node("redirect_node", redirect_node)
    graph.add_node("decline_node", decline_node)
    graph.add_node("resume_qa_node", resume_qa_node)
    graph.add_node("generate_feedback_node", generate_feedback_node)
    graph.add_node("wait_for_role_input_node", wait_for_role_input_node)
    graph.add_node("wait_for_question_retry_node", wait_for_question_retry_node)

    graph.set_conditional_entry_point(
        route_at_start,
        {
            "role_intake_node": "role_intake_node",
            "classify_answer_node": "classify_answer_node",
            "generate_feedback_node": "generate_feedback_node",
        },
    )

    graph.add_conditional_edges(
        "role_intake_node",
        route_after_role_intake,
        {
            "wait_for_role_input_node": "wait_for_role_input_node",
            "ask_question_node": "ask_question_node",
        },
    )
    graph.add_edge("wait_for_role_input_node", "role_intake_node")

    graph.add_conditional_edges(
        "ask_question_node",
        route_after_ask_question,
        {
            "wait_for_question_retry_node": "wait_for_question_retry_node",
            "classify_answer_node": "classify_answer_node",
        },
    )
    graph.add_edge("wait_for_question_retry_node", "ask_question_node")

    graph.add_conditional_edges(
        "classify_answer_node",
        route_after_classify,
        {
            "follow_up_node": "follow_up_node",
            "next_question_node": "next_question_node",
            "redirect_node": "redirect_node",
            "decline_node": "decline_node",
            "resume_qa_node": "resume_qa_node",
        },
    )

    graph.add_conditional_edges(
        "next_question_node",
        route_after_next_question,
        {
            "ask_question_node": "ask_question_node",
            "generate_feedback_node": "generate_feedback_node",
        },
    )

    graph.add_edge("follow_up_node", "classify_answer_node")
    graph.add_edge("redirect_node", "classify_answer_node")
    graph.add_edge("decline_node", "classify_answer_node")
    # resume_qa_node answers the question and hands back to classify_answer_node,
    # which is an interrupt point — the turn ends and the interview resumes
    # normally on the next user message.
    graph.add_edge("resume_qa_node", "classify_answer_node")
    graph.add_edge("generate_feedback_node", END)

    if checkpointer is None:
        db_path = os.getenv("LANGGRAPH_CHECKPOINT_DB", _DEFAULT_CHECKPOINT_DB)
        checkpointer, _conn = _make_sqlite_checkpointer(db_path)
        global _checkpointer_conn
        _checkpointer_conn = _conn

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "wait_for_role_input_node",
            "wait_for_question_retry_node",
            "classify_answer_node",
        ],
    )
    logger.info("LangGraph interview graph compiled successfully.")
    return compiled


def get_interview_graph():
    """Process-wide compiled graph with a durable SQLite checkpointer."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_interview_graph()
    return _compiled_graph


def reset_interview_graph_cache() -> None:
    """Test helper to drop the cached compiled graph."""
    global _compiled_graph
    _compiled_graph = None
