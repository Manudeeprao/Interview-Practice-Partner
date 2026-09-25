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
