"""
Flask application for the Interview Practice Partner backend.

Routes:
  POST /chat      — Accepts user message + session state, runs LangGraph, returns agent response
  POST /feedback  — Accepts full transcript, runs generate_feedback_node, returns structured JSON
  GET  /health    — Health check endpoint

Architecture:
  Per-session state is stored in memory (dict keyed by session_id).
  The LangGraph graph is compiled once at startup and invoked per-turn.
  
  Because the interview pauses between turns for user input, we don't run the
  full graph end-to-end in one call. Instead, each POST /chat runs the graph
  starting from the appropriate entry node (determined by session state), 
  advances until the next user-input pause, and returns the agent's response.
"""

import os
import logging
import uuid
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

from graph.nodes import (
    role_intake_node,
    ask_question_node,
    classify_answer_node,
    follow_up_node,
    next_question_node,
    redirect_node,
    decline_node,
    generate_feedback_node,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow all origins for local development

# ---------------------------------------------------------------------------
# In-memory session store
# Format: { session_id: InterviewState }
# ---------------------------------------------------------------------------
sessions: dict = {}

MAX_SESSIONS = 100  # simple cap to avoid unbounded memory growth


def create_initial_state(session_id: str) -> dict:
    """Create a fresh InterviewState for a new session."""
    return {
        "session_id": session_id,
        "role": None,
        "role_confirmed": False,
        "main_question_count": 0,
        "follow_up_count": 0,
        "is_follow_up": False,
        "max_questions": 7,  # Fixed to 7 for balanced interview: intro(1) + project(2) + technical(2) + system_design(1) + behavioral(1)
        "interview_stage": STAGE_ROLE_SELECTION,
        "previous_questions": [],
        "user_answers": [],
        "current_question": None,
        "history": [],
        "last_user_message": "",
        "classification": None,
        "classification_reason": None,
        "agent_response": None,
        "next_node": "role_intake_node",
        "project_question_count": 0,  # Tracks project-related questions (max 2)
        "feedback": None,
        "done": False,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node dispatcher — manually routes to the right node based on state
# ---------------------------------------------------------------------------
NODE_MAP = {
    "role_intake_node": role_intake_node,
    "ask_question_node": ask_question_node,
    "classify_answer_node": classify_answer_node,
    "follow_up_node": follow_up_node,
    "next_question_node": next_question_node,
    "redirect_node": redirect_node,
    "decline_node": decline_node,
    "generate_feedback_node": generate_feedback_node,
}


# ---------------------------------------------------------------------------
# Interview stage constants
# ---------------------------------------------------------------------------
STAGE_ROLE_SELECTION    = "ROLE_SELECTION"
STAGE_INTERVIEW_ACTIVE  = "INTERVIEW_ACTIVE"
STAGE_FEEDBACK          = "FEEDBACK_GENERATION"
STAGE_COMPLETE          = "INTERVIEW_COMPLETE"

# Nodes that are legal only during ROLE_SELECTION stage
ROLE_SELECTION_NODES = {"role_intake_node"}

# Nodes that are legal during INTERVIEW_ACTIVE stage
INTERVIEW_ACTIVE_NODES = {
    "classify_answer_node", "follow_up_node",
    "next_question_node", "redirect_node",
    "decline_node", "ask_question_node",
    "generate_feedback_node",
}


def run_turn(state: dict, user_message: str) -> dict:
    """
    Run one conversation turn through the LangGraph nodes.

    Stage-aware dispatch:
      - ROLE_SELECTION:   only role_intake_node is allowed
      - INTERVIEW_ACTIVE: role_intake_node is FORBIDDEN; routes through
                          classify -> follow_up / next_question / redirect / decline
      - FEEDBACK_GENERATION / INTERVIEW_COMPLETE: handled by generate_feedback_node
    """
    state = dict(state)  # shallow copy
    state["last_user_message"] = user_message

    role_confirmed  = state.get("role_confirmed", False)
    interview_stage = state.get("interview_stage", STAGE_ROLE_SELECTION)
    next_node       = state.get("next_node", "role_intake_node")

    # ── Stage guard ───────────────────────────────────────────────────────
    # Once the interview is active, NEVER route back to role_intake_node.
    # This prevents user interview answers containing role keywords (e.g.
    # "I worked as a data analyst") from being mis-detected as role selection.
    if role_confirmed and next_node in ROLE_SELECTION_NODES:
        logger.warning(
            "[STAGE GUARD] interview_stage=%s, role_confirmed=True but next_node=%s. "
            "Forcing classify_answer_node to prevent role re-detection.",
            interview_stage, next_node,
        )
        next_node = "classify_answer_node"
        state["next_node"] = "classify_answer_node"

    # Ensure interview_stage is correct for active interviews
    if role_confirmed and interview_stage == STAGE_ROLE_SELECTION:
        state["interview_stage"] = STAGE_INTERVIEW_ACTIVE
        interview_stage = STAGE_INTERVIEW_ACTIVE

    # ── Debug header log ─────────────────────────────────────────────────
    logger.info(
        "[TURN START] Stage=%s | Role=%s | Q=%d/%d | StartNode=%s | Msg='%s'",
        interview_stage,
        state.get("role", "(none)"),
        state.get("main_question_count", 0),
        state.get("max_questions", 6),
        next_node,
        user_message[:60],
    )

    MAX_AUTO_STEPS = 6
    steps = 0

    while steps < MAX_AUTO_STEPS:
        steps += 1
        node_fn = NODE_MAP.get(next_node)

        if not node_fn:
            # Safe fallback: never route to role_intake_node if interview is active
            if role_confirmed:
                logger.error(
                    "Unknown node '%s' during active interview — "
                    "falling back to classify_answer_node (NOT role_intake_node).",
                    next_node,
                )
                node_fn = classify_answer_node
                next_node = "classify_answer_node"
            else:
                logger.error(
                    "Unknown node '%s' before role confirmed — "
                    "falling back to role_intake_node.",
                    next_node,
                )
                node_fn = role_intake_node
                next_node = "role_intake_node"

        logger.info(
            "[NODE] Running %s (step %d) | Stage=%s | Q=%d",
            next_node, steps, state.get("interview_stage", "?"),
            state.get("main_question_count", 0),
        )
        current_node = next_node
        updates = node_fn(state)
        state.update(updates)
        next_node = state.get("next_node", "")

        # Update role_confirmed from state in case it just got set this step
        role_confirmed = state.get("role_confirmed", False)

        # Sync interview_stage: as soon as role is confirmed, mark INTERVIEW_ACTIVE
        if role_confirmed and state.get("interview_stage") == STAGE_ROLE_SELECTION:
            state["interview_stage"] = STAGE_INTERVIEW_ACTIVE

        # Stop conditions: agent has a response ready for the user
        if next_node in ("classify_answer_node", "__end__", "") or state.get("done"):
            break

        # Self-loop detection
        if next_node == current_node:
            logger.info("[LOOP] Self-loop at %s — returning response to user.", current_node)
            break

    # ── Debug footer log ─────────────────────────────────────────────────
    logger.info(
        "[TURN END] Stage=%s | Q=%d/%d | Classification=%s | NextNode=%s | Done=%s",
        state.get("interview_stage", "?"),
        state.get("main_question_count", 0),
        state.get("max_questions", 6),
        state.get("classification", "N/A"),
        state.get("next_node", "???"),
        state.get("done", False),
    )

    return state


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "sessions": len(sessions)})


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Request body:
      {
        "session_id": "optional-uuid",
        "message": "user's text input"
      }
    
    Response:
      {
        "session_id": "...",
        "response": "agent's response text",
        "done": false,
        "feedback": null,
        "state_info": { "role": "...", "main_question_count": 3, "node": "..." }
      }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "message field is required"}), 400

        session_id = data.get("session_id") or str(uuid.uuid4())

        # Retrieve or create session
        if session_id not in sessions:
            # Clean up oldest session if at cap
            if len(sessions) >= MAX_SESSIONS:
                oldest_key = next(iter(sessions))
                del sessions[oldest_key]
                logger.info("Evicted oldest session: %s", oldest_key)
            sessions[session_id] = create_initial_state(session_id)
            logger.info("New session created: %s", session_id)

        state = sessions[session_id]

        # Don't accept new messages if interview is done
        if state.get("done"):
            return jsonify({
                "session_id": session_id,
                "response": "The interview has ended. Please refresh to start a new session.",
                "done": True,
                "feedback": state.get("feedback"),
                "state_info": _state_info(state),
            })

        # Run the turn through the graph
        updated_state = run_turn(state, user_message)
        sessions[session_id] = updated_state

        agent_response = updated_state.get("agent_response", "I'm here — please go ahead.")

        # ── Per-turn debug log ────────────────────────────────────────────────
        logger.info(
            ">>> TURN RESULT: classification=%s, main_question_count=%d/%d, "
            "follow_up_count=%d, is_follow_up=%s, next_node=%s",
            updated_state.get("classification", "N/A"),
            updated_state.get("main_question_count", 0),
            updated_state.get("max_questions", 6),
            updated_state.get("follow_up_count", 0),
            updated_state.get("is_follow_up", False),
            updated_state.get("next_node", "???"),
        )

        return jsonify({
            "session_id": session_id,
            "response": agent_response,
            "done": updated_state.get("done", False),
            "feedback": updated_state.get("feedback"),
            "state_info": _state_info(updated_state),
        })

    except Exception as e:
        logger.exception("Error in /chat endpoint: %s", e)
        return jsonify({
            "error": "An internal error occurred. Please try again.",
            "session_id": data.get("session_id", "") if data else "",
            "response": "I'm sorry, I hit a technical glitch. Could you repeat that?",
            "done": False,
        }), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    POST /feedback
    Directly triggers generate_feedback_node on the full transcript.
    
    Request body:
      {
        "session_id": "...",
        "transcript": [{"role": "user/assistant", "content": "..."}]
      }
    
    Response:
      {
        "feedback": {
          "overallImpression": "...",
          "communication": "...",
          "technicalKnowledge": "...",
          "improvementAreas": ["...", "..."]
        }
      }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id", "")
        transcript = data.get("transcript", [])

        # Use session history if available, fallback to provided transcript
        state = sessions.get(session_id)
        if state:
            # Mark done and run feedback node
            state["done"] = False  # temporarily allow feedback generation
            feedback_updates = generate_feedback_node(state)
            sessions[session_id] = {**state, **feedback_updates}
            return jsonify({"feedback": feedback_updates.get("feedback", {})})
        else:
            # Build minimal state from provided transcript
            minimal_state = create_initial_state(session_id or "feedback-only")
            minimal_state["history"] = transcript
            minimal_state["role"] = data.get("role", "General")
            feedback_updates = generate_feedback_node(minimal_state)
            return jsonify({"feedback": feedback_updates.get("feedback", {})})

    except Exception as e:
        logger.exception("Error in /feedback endpoint: %s", e)
        return jsonify({"error": "Failed to generate feedback. Please try again."}), 500


@app.route("/new_session", methods=["POST"])
def new_session():
    """Create a fresh session and return the new session_id."""
    session_id = str(uuid.uuid4())
    if len(sessions) >= MAX_SESSIONS:
        oldest_key = next(iter(sessions))
        del sessions[oldest_key]
    sessions[session_id] = create_initial_state(session_id)
    return jsonify({"session_id": session_id})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _state_info(state: dict) -> dict:
    """Extract minimal state info to send to the frontend for UI state management."""
    return {
        "role": state.get("role"),
        "role_confirmed": state.get("role_confirmed", False),
        "main_question_count": state.get("main_question_count", 0),
        "follow_up_count": state.get("follow_up_count", 0),
        "max_questions": state.get("max_questions", 6),
        "interview_stage": state.get("interview_stage", "introduction"),
        "current_node": state.get("next_node", "role_intake_node"),
        "classification": state.get("classification"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    logger.info("Starting Interview Practice Partner backend on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
