from graph.state import create_initial_state
from storage import SessionStore


def test_round_trip_persists_interview_state(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.sqlite"))
    state = create_initial_state("persist-1")
    state["role"] = "Software Engineer"
    state["role_confirmed"] = True
    state["difficulty"] = "hard"
    store.save(state)

    loaded = store.get("persist-1")
    assert loaded is not None
    assert loaded["role"] == "Software Engineer"
    assert loaded["role_confirmed"] is True
    assert loaded["difficulty"] == "hard"
    assert loaded["history"] == []


def test_missing_session_returns_none(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.sqlite"))
    assert store.get("missing") is None


def test_updates_overwrite_json_blob(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.sqlite"))
    state = create_initial_state("persist-2")
    store.save(state)
    state["main_question_count"] = 3
    store.save(state)
    assert store.get("persist-2")["main_question_count"] == 3


def test_evicts_oldest_sessions(tmp_path):
    store = SessionStore(db_path=str(tmp_path / "sessions.sqlite"), max_sessions=2)
    store.save(create_initial_state("a"))
    store.save(create_initial_state("b"))
    store.save(create_initial_state("c"))
    assert store.count() == 2
    assert store.get("a") is None
    assert store.get("c") is not None
