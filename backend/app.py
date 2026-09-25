"""
Flask application for the Interview Practice Partner backend.

Each POST /chat resumes the compiled LangGraph checkpoint for that session.
InterviewState is persisted as JSON in SQLite.
"""

from __future__ import annotations

import logging
import os
import uuid

from flask import Flask, current_app, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from graph.graph import get_interview_graph
from graph.nodes import generate_feedback_node
from graph.state import STAGE_INTERVIEW_ACTIVE, STAGE_ROLE_SELECTION, create_initial_state
from rag.retriever import ResumeRetriever
from storage import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SESSIONS = 100
MAX_RESUME_BYTES = 5 * 1024 * 1024


def _checkpoint_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _snapshot_has_values(snapshot) -> bool:
    if snapshot is None:
        return False
    values = getattr(snapshot, "values", None)
    return bool(values)


def run_turn(state: dict, user_message: str, graph=None) -> dict:
    """Resume the compiled interview graph for one HTTP conversation turn."""
    graph = graph or current_app.config["INTERVIEW_GRAPH"]
    state = dict(state)
    state["last_user_message"] = user_message
    session_id = state["session_id"]
    config = _checkpoint_config(session_id)

    logger.info(
        "[TURN START] Stage=%s | Role=%s | Q=%d/%d | Difficulty=%s | Msg='%s'",
        state.get("interview_stage", STAGE_ROLE_SELECTION),
        state.get("role", "(none)"),
        state.get("main_question_count", 0),
        state.get("max_questions", 7),
        state.get("difficulty", "medium"),
        user_message[:60],
    )

    snapshot = graph.get_state(config)
    if _snapshot_has_values(snapshot):
        graph.update_state(config, {"last_user_message": user_message})
        result = graph.invoke(None, config=config)
    else:
        result = graph.invoke(state, config=config)

    if not isinstance(result, dict) or not result:
        latest = graph.get_state(config)
        result = dict(getattr(latest, "values", {}) or state)

    merged = {**state, **result}
    if merged.get("role_confirmed") and merged.get("interview_stage") == STAGE_ROLE_SELECTION:
        merged["interview_stage"] = STAGE_INTERVIEW_ACTIVE

    logger.info(
        "[TURN END] Stage=%s | Q=%d/%d | Classification=%s | Difficulty=%s | NextNode=%s | Done=%s",
        merged.get("interview_stage", "?"),
        merged.get("main_question_count", 0),
        merged.get("max_questions", 7),
        merged.get("classification", "N/A"),
        merged.get("difficulty", "medium"),
        merged.get("next_node", "???"),
        merged.get("done", False),
    )
    return merged


def create_app(
    session_store: SessionStore | None = None,
    interview_graph=None,
    resume_retriever: ResumeRetriever | None = None,
) -> Flask:
    app = Flask(__name__)
    CORS(app)

    app.config["SESSION_STORE"] = session_store or SessionStore(max_sessions=MAX_SESSIONS)
    app.config["INTERVIEW_GRAPH"] = interview_graph or get_interview_graph()
    app.config["RESUME_RETRIEVER"] = resume_retriever or ResumeRetriever()

    register_routes(app)
    return app


def register_routes(app: Flask) -> None:
    @app.route("/health", methods=["GET"])
    def health():
        store: SessionStore = current_app.config["SESSION_STORE"]
        return jsonify({"status": "ok", "sessions": store.count()})

    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            user_message = data.get("message", "").strip()
            if not user_message:
                return jsonify({"error": "message field is required"}), 400

            session_id = data.get("session_id") or str(uuid.uuid4())
            store: SessionStore = current_app.config["SESSION_STORE"]
            store.ensure_capacity()

            state = store.get(session_id)
            if state is None:
                state = create_initial_state(session_id)
                store.save(state)
                logger.info("New session created: %s", session_id)

            if state.get("done"):
                return jsonify({
                    "session_id": session_id,
                    "response": "The interview has ended. Please refresh to start a new session.",
                    "done": True,
                    "feedback": state.get("feedback"),
                    "state_info": _state_info(state),
                })

            updated_state = run_turn(state, user_message)
            store.save(updated_state)

            agent_response = updated_state.get("agent_response", "I'm here — please go ahead.")
            logger.info(
                ">>> TURN RESULT: classification=%s, strength=%s, difficulty=%s, "
                "main_question_count=%d/%d, next_node=%s",
                updated_state.get("classification", "N/A"),
                updated_state.get("answer_strength", "N/A"),
                updated_state.get("difficulty", "medium"),
                updated_state.get("main_question_count", 0),
                updated_state.get("max_questions", 7),
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
            payload = data if "data" in locals() else {}
            return jsonify({
                "error": "An internal error occurred. Please try again.",
                "session_id": payload.get("session_id", "") if payload else "",
                "response": "I'm sorry, I hit a technical glitch. Could you repeat that?",
                "done": False,
            }), 500

    @app.route("/feedback", methods=["POST"])
    def feedback():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            session_id = data.get("session_id", "")
            transcript = data.get("transcript", [])
            store: SessionStore = current_app.config["SESSION_STORE"]

            state = store.get(session_id) if session_id else None
            if state:
                state["done"] = False
                feedback_updates = generate_feedback_node(state)
                updated = {**state, **feedback_updates}
                store.save(updated)
                return jsonify({"feedback": feedback_updates.get("feedback", {})})

            minimal_state = create_initial_state(session_id or "feedback-only")
            minimal_state["history"] = transcript
            minimal_state["role"] = data.get("role", "General")
            feedback_updates = generate_feedback_node(minimal_state)
            return jsonify({"feedback": feedback_updates.get("feedback", {})})

        except Exception as e:
            logger.exception("Error in /feedback endpoint: %s", e)
            return jsonify({"error": "Failed to generate feedback. Please try again."}), 500

    @app.route("/resume/upload", methods=["POST"])
    def upload_resume():
        try:
            if "file" not in request.files:
                return jsonify({"error": "Please include a resume file in the 'file' field."}), 400

            uploaded_file = request.files["file"]
            if uploaded_file.filename == "":
                return jsonify({"error": "No file selected."}), 400

            session_id = request.form.get("session_id") or str(uuid.uuid4())
            store: SessionStore = current_app.config["SESSION_STORE"]
            retriever: ResumeRetriever = current_app.config["RESUME_RETRIEVER"]
            store.ensure_capacity()

            state = store.get(session_id)
            if state is None:
                state = create_initial_state(session_id)
                store.save(state)

            file_bytes = uploaded_file.read()
            if len(file_bytes) > MAX_RESUME_BYTES:
                return jsonify({"error": "Resume file is too large. Please keep it under 5 MB."}), 413

            filename = uploaded_file.filename or "resume"
            suffix = os.path.splitext(filename)[1].lower()
            if suffix not in {".pdf", ".docx"}:
                return jsonify({"error": "Only PDF and DOCX resumes are supported."}), 400

            from rag.extractor import extract_resume_text

            try:
                text = extract_resume_text(file_bytes, filename)
            except ValueError as exc:
                logger.warning("Resume upload rejected: %s", exc)
                return jsonify({"error": str(exc)}), 400

            if not text.strip():
                return jsonify({"error": "The uploaded resume did not contain any readable text."}), 400

            retriever.vector_store.delete_session(session_id)
            indexed_chunks = retriever.index_resume(
                session_id=session_id,
                text=text,
                metadata={"filename": filename},
            )

            state["resume_uploaded"] = True
            state["resume_filename"] = filename
            state["resume_chunk_count"] = len(indexed_chunks)
            store.save(state)

            graph = current_app.config["INTERVIEW_GRAPH"]
            config = _checkpoint_config(session_id)
            if _snapshot_has_values(graph.get_state(config)):
                graph.update_state(
                    config,
                    {
                        "resume_uploaded": True,
                        "resume_filename": filename,
                        "resume_chunk_count": len(indexed_chunks),
                    },
                )

            return jsonify({
                "session_id": session_id,
                "message": "Resume uploaded successfully.",
                "filename": filename,
                "chunks": len(indexed_chunks),
            })
        except ValueError as exc:
            logger.warning("Resume upload rejected: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Resume upload failed: %s", exc)
            return jsonify({"error": "Failed to process the uploaded resume."}), 500

    @app.route("/new_session", methods=["POST"])
    def new_session():
        store: SessionStore = current_app.config["SESSION_STORE"]
        store.ensure_capacity()
        session_id = str(uuid.uuid4())
        store.save(create_initial_state(session_id))
        return jsonify({"session_id": session_id})


def _state_info(state: dict) -> dict:
    return {
        "role": state.get("role"),
        "role_confirmed": state.get("role_confirmed", False),
        "main_question_count": state.get("main_question_count", 0),
        "follow_up_count": state.get("follow_up_count", 0),
        "max_questions": state.get("max_questions", 7),
        "interview_stage": state.get("interview_stage", "introduction"),
        "current_node": state.get("next_node", "role_intake_node"),
        "classification": state.get("classification"),
        "difficulty": state.get("difficulty", "medium"),
        "answer_strength": state.get("answer_strength"),
        "resume_uploaded": state.get("resume_uploaded", False),
    }


app = None if os.getenv("IPP_TESTING") == "1" else create_app()


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    logger.info("Starting Interview Practice Partner backend on port %d", port)
    application = app or create_app()
    application.run(host="0.0.0.0", port=port, debug=debug)
