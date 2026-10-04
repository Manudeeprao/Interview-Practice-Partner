"""SQLite persistence for InterviewState (JSON per session)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "data")
_DEFAULT_DB = os.path.join(_DEFAULT_DIR, "sessions.sqlite")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Persist InterviewState dicts as JSON in SQLite.

    When the store exceeds ``max_sessions``, the oldest sessions are evicted.
    Pass ``on_evict`` to clean up per-session resources elsewhere (e.g. the
    session's ChromaDB resume vectors) when that happens.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_sessions: int = 100,
        on_evict: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.db_path = db_path or os.getenv("SESSION_DB_PATH", _DEFAULT_DB)
        self.max_sessions = max_sessions
        self.on_evict = on_evict
        self._lock = threading.Lock()
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["state_json"])

    def save(self, state: Dict[str, Any]) -> None:
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("Cannot persist interview state without session_id")
        payload = json.dumps(state)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, state_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, payload, _utc_now()),
                )
            self._evict_if_needed()

    def delete(self, session_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return int(row["n"] if row else 0)

    def ensure_capacity(self) -> None:
        with self._lock:
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        n = int(row["n"] if row else 0)
        overflow = n - self.max_sessions
        if overflow <= 0:
            return
        victims = self._conn.execute(
            "SELECT session_id FROM sessions ORDER BY updated_at ASC LIMIT ?",
            (overflow,),
        ).fetchall()
        ids = [v["session_id"] for v in victims]
        self._conn.executemany(
            "DELETE FROM sessions WHERE session_id = ?",
            [(sid,) for sid in ids],
        )
        logger.info("Evicted %d oldest session(s): %s", len(ids), ids)
        if self.on_evict:
            for sid in ids:
                try:
                    self.on_evict(sid)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("on_evict failed for session %s: %s", sid, exc)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
