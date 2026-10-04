"""
Flask application for the Interview Practice Partner backend.

Each POST /chat resumes the compiled LangGraph checkpoint for that session.
InterviewState is persisted as JSON in SQLite; resume vectors live in ChromaDB
and are cleaned up when a session is deleted, reset, or evicted.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from typing import Optional

from flask import Flask, current_app, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from graph.graph import get_interview_graph
from graph.nodes import configure_resume_retriever, generate_feedback_node
from graph.state import STAGE_INTERVIEW_ACTIVE, STAGE_ROLE_SELECTION, create_initial_state
from rag.retriever import ResumeRetriever
from storage import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))
MAX_RESUME_BYTES = 5 * 1024 * 1024
MAX_MESSAGE_CHARS = 4000

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error(message: str, code: str, status: int):
    """Consistent JSON error shape: {error, code}."""
    resp = jsonify({"error": message, "code": code})
    resp.status_code = status
    return resp


def _valid_session_id(session_id: Optional[str]) -> bool:
    return bool(session_id) and bool(_SESSION_ID_RE.match(session_id))


class RateLimiter:
    """Tiny in-memory sliding-window rate limiter (per key, thread-safe)."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits[key] if now - t < self.window]
            if len(hits) >= self.max_requests:
                retry_after = int(self.window - (now - hits[0])) + 1
                self._hits[key] = hits
                return False, retry_after
            hits.append(now)
            self._hits[key] = hits
            return True, 0


def _client_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.remote_addr or "unknown"


def _rate_limited(limiter: RateLimiter):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            allowed, retry_after = limiter.allow(_client_key())
            if not allowed:
                resp = _error(
                    "Too many requests. Please slow down and try again.",
                    "rate_limited",
                    429,
                )
                resp.headers["Retry-After"] = str(retry_after)
                return resp
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _checkpoint_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _snapshot_has_values(snapshot) -> bool:
    if snapshot is None:
        return False
    values = getattr(snapshot, "values", None)
    return bool(values)


def _delete_checkpoint_thread(graph, session_id: str) -> None:
    """Best-effort removal of a session's LangGraph checkpoints."""
    try:
        checkpointer = getattr(graph, "checkpointer", None)
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if callable(delete_thread):
            delete_thread(session_id)
            logger.info("Deleted LangGraph checkpoint thread for %s", session_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not delete checkpoint thread %s: %s", session_id, exc)


def _reset_session(session_id: str, store: SessionStore, retriever: ResumeRetriever, graph) -> None:
    """Wipe a session everywhere: SQLite state, Chroma vectors, checkpoints."""
    store.delete(session_id)
    try:
        retriever.vector_store.delete_session(session_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Resume cleanup failed for %s: %s", session_id, exc)
    _delete_checkpoint_thread(graph, session_id)
    logger.info("Session reset: %s", session_id)


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
        graph.update_state(
            config,
            {
                "last_user_message": user_message,
                "resume_uploaded": state.get("resume_uploaded", False),
                "resume_filename": state.get("resume_filename"),
                "resume_chunk_count": state.get("resume_chunk_count", 0),
            },
        )
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

    cors_origins = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
        if o.strip()
    ]
    CORS(app, origins=cors_origins)

    retriever = resume_retriever or ResumeRetriever()
    if session_store is None:
        # Evicting a session also purges its resume vectors from ChromaDB.
        session_store = SessionStore(
            max_sessions=MAX_SESSIONS,
            on_evict=retriever.vector_store.delete_session,
        )
    app.config["SESSION_STORE"] = session_store
    app.config["INTERVIEW_GRAPH"] = interview_graph or get_interview_graph()
    app.config["RESUME_RETRIEVER"] = retriever
    # Read at app-creation time so tests can override via environment.
    app.config["CHAT_LIMITER"] = RateLimiter(
        int(os.getenv("RATE_LIMIT_CHAT_PER_MIN", "30"))
    )
    app.config["UPLOAD_LIMITER"] = RateLimiter(
        int(os.getenv("RATE_LIMIT_UPLOAD_PER_MIN", "10"))
    )
    configure_resume_retriever(retriever)

    register_routes(app)
    return app


def register_routes(app: Flask) -> None:
    chat_limited = _rate_limited(app.config["CHAT_LIMITER"])
    upload_limited = _rate_limited(app.config["UPLOAD_LIMITER"])

    @app.route("/health", methods=["GET"])
    def health():
        store: SessionStore = current_app.config["SESSION_STORE"]
        return jsonify({"status": "ok", "sessions": store.count()})

    @app.route("/chat", methods=["POST"])
    @chat_limited
    def chat():
        try:
            data = request.get_json(silent=True)
            if not data:
                return _error("Request body must be JSON.", "invalid_body", 400)

            user_message = data.get("message", "")
            if not isinstance(user_message, str) or not user_message.strip():
                return _error("The 'message' field is required.", "missing_message", 400)
            user_message = user_message.strip()
            if len(user_message) > MAX_MESSAGE_CHARS:
                return _error(
                    f"Message too long (max {MAX_MESSAGE_CHARS} characters).",
                    "message_too_long",
                    400,
                )

            session_id = data.get("session_id") or str(uuid.uuid4())
            if not _valid_session_id(session_id):
                return _error("Invalid session_id format.", "invalid_session_id", 400)

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
                    "response": "The interview has ended. Please start a new session to practice again.",
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
            payload = data if "data" in locals() and isinstance(data, dict) else {}
            resp = jsonify({
                "error": "An internal error occurred. Please try again.",
                "code": "internal_error",
                "session_id": payload.get("session_id", ""),
                "response": "I'm sorry, I hit a technical glitch. Could you repeat that?",
                "done": False,
            })
            resp.status_code = 500
            return resp

    @app.route("/feedback", methods=["POST"])
    def feedback():
        try:
            data = request.get_json(silent=True)
            if not data:
                return _error("Request body must be JSON.", "invalid_body", 400)

            session_id = data.get("session_id", "")
            if session_id and not _valid_session_id(session_id):
                return _error("Invalid session_id format.", "invalid_session_id", 400)
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
            return _error("Failed to generate feedback. Please try again.", "feedback_failed", 500)

    @app.route("/session/<session_id>", methods=["GET"])
    def get_session(session_id: str):
        """Return a session's transcript and progress (for page-refresh restore)."""
        if not _valid_session_id(session_id):
            return _error("Invalid session_id format.", "invalid_session_id", 400)
        store: SessionStore = current_app.config["SESSION_STORE"]
        state = store.get(session_id)
        if state is None:
            return _error("Session not found.", "session_not_found", 404)
        history = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in state.get("history", [])
            if isinstance(m, dict)
        ]
        return jsonify({
            "session_id": session_id,
            "history": history,
            "state_info": _state_info(state),
            "done": state.get("done", False),
            "feedback": state.get("feedback"),
        })

    @app.route("/resume/upload", methods=["POST"])
    @upload_limited
    def upload_resume():
        try:
            if "file" not in request.files:
                return _error("Please include a resume file in the 'file' field.", "missing_file", 400)

            uploaded_file = request.files["file"]
            if uploaded_file.filename == "":
                return _error("No file selected.", "missing_file", 400)

            session_id = request.form.get("session_id") or str(uuid.uuid4())
            if not _valid_session_id(session_id):
                return _error("Invalid session_id format.", "invalid_session_id", 400)

            store: SessionStore = current_app.config["SESSION_STORE"]
            retriever: ResumeRetriever = current_app.config["RESUME_RETRIEVER"]
            store.ensure_capacity()

            state = store.get(session_id)
            if state is None:
                state = create_initial_state(session_id)
                store.save(state)

            file_bytes = uploaded_file.read()
            if len(file_bytes) > MAX_RESUME_BYTES:
                return _error("Resume file is too large. Please keep it under 5 MB.", "file_too_large", 413)

            filename = uploaded_file.filename or "resume"
            suffix = os.path.splitext(filename)[1].lower()
            if suffix not in {".pdf", ".docx"}:
                return _error("Only PDF and DOCX resumes are supported.", "unsupported_file_type", 400)

            # Verify the content matches the extension (magic bytes), not just the name.
            if suffix == ".pdf" and not file_bytes.lstrip().startswith(b"%PDF"):
                return _error("The file does not look like a valid PDF.", "invalid_file_content", 400)
            if suffix == ".docx" and not file_bytes.startswith(b"PK\x03\x04"):
                return _error("The file does not look like a valid DOCX.", "invalid_file_content", 400)

            from rag.extractor import extract_resume_text

            try:
                text = extract_resume_text(file_bytes, filename)
            except ValueError as exc:
                logger.warning("Resume upload rejected: %s", exc)
                return _error(str(exc), "extraction_failed", 400)

            if not text.strip():
                return _error("The uploaded resume did not contain any readable text.", "empty_resume", 400)

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
            return _error(str(exc), "invalid_request", 400)
        except Exception as exc:
            logger.exception("Resume upload failed: %s", exc)
            return _error("Failed to process the uploaded resume.", "upload_failed", 500)

    @app.route("/resume/status", methods=["GET"])
    def resume_status():
        """Return a session's resume state: {uploaded, filename, chunks}."""
        session_id = request.args.get("session_id", "")
        if not _valid_session_id(session_id):
            return _error("Invalid session_id format.", "invalid_session_id", 400)
        store: SessionStore = current_app.config["SESSION_STORE"]
        retriever: ResumeRetriever = current_app.config["RESUME_RETRIEVER"]
        state = store.get(session_id)
        chunks = retriever.vector_store.count_session(session_id)
        uploaded = chunks > 0 or bool(state and state.get("resume_uploaded"))
        return jsonify({
            "session_id": session_id,
            "uploaded": uploaded,
            "filename": (state or {}).get("resume_filename"),
            "chunks": chunks,
        })

    @app.route("/resume/<session_id>", methods=["DELETE"])
    def delete_resume(session_id: str):
        """Remove a session's resume vectors and clear its resume flags."""
        if not _valid_session_id(session_id):
            return _error("Invalid session_id format.", "invalid_session_id", 400)
        store: SessionStore = current_app.config["SESSION_STORE"]
        retriever: ResumeRetriever = current_app.config["RESUME_RETRIEVER"]

        retriever.vector_store.delete_session(session_id)

        state = store.get(session_id)
        if state is not None:
            state["resume_uploaded"] = False
            state["resume_filename"] = None
            state["resume_chunk_count"] = 0
            store.save(state)
            graph = current_app.config["INTERVIEW_GRAPH"]
            config = _checkpoint_config(session_id)
            if _snapshot_has_values(graph.get_state(config)):
                graph.update_state(
                    config,
                    {
                        "resume_uploaded": False,
                        "resume_filename": None,
                        "resume_chunk_count": 0,
                    },
                )

        return jsonify({
            "session_id": session_id,
            "deleted": True,
            "message": "Resume removed.",
        })

    @app.route("/new_session", methods=["POST"])
    def new_session():
        """Create a session, or reset an existing one (state + checkpoints + resume)."""
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")
        store: SessionStore = current_app.config["SESSION_STORE"]
        retriever: ResumeRetriever = current_app.config["RESUME_RETRIEVER"]
        graph = current_app.config["INTERVIEW_GRAPH"]

        if session_id:
            if not _valid_session_id(session_id):
                return _error("Invalid session_id format.", "invalid_session_id", 400)
            _reset_session(session_id, store, retriever, graph)
        else:
            store.ensure_capacity()
            session_id = str(uuid.uuid4())

        store.save(create_initial_state(session_id))
        logger.info("New session created: %s", session_id)
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
