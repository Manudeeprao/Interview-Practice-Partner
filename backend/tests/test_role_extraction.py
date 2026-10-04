from graph.nodes import _extract_role_from_text


def test_extracts_common_software_role():
    assert _extract_role_from_text("I want to practice for a software engineer role") == "Software Engineer"


def test_extracts_data_scientist():
    assert _extract_role_from_text("Please interview me as a data scientist") == "Data Scientist"


def test_extracts_role_from_apply_pattern():
    role = _extract_role_from_text("I am applying for product designer")
    assert role is not None
    assert "product" in role.lower()


def test_returns_none_when_role_is_unclear():
    assert _extract_role_from_text("hello there") is None


# --- Regression tests: live end-to-end run 2026-10-04 ---------------------
def test_extracts_backend_engineer_bare():
    # The most natural single-phrase input must not fall through to the LLM.
    assert _extract_role_from_text("backend engineer") == "Backend Engineer"


def test_extracts_frontend_engineer_bare():
    assert _extract_role_from_text("frontend engineer") == "Frontend Engineer"


def test_strips_leading_article():
    assert _extract_role_from_text("I want to interview for the backend engineer role") == "Backend Engineer"


def test_long_answer_text_does_not_yield_bogus_role():
    # "Redis for caching hot product data" must NOT become "Caching Hot Product Data".
    answer = (
        "I would use a layered architecture: stateless API servers behind a load "
        "balancer, Redis for caching hot product data, PostgreSQL with read replicas "
        "for persistence, and Kafka for async order processing. Rate limiting at the "
        "gateway, and idempotency keys on order creation to handle retries safely."
    )
    assert _extract_role_from_text(answer) is None


def test_long_message_with_for_phrase_is_ignored():
    assert _extract_role_from_text(
        "In my last job I built dashboards for tracking user engagement metrics "
        "across three product lines and presented findings to stakeholders weekly."
    ) is None
