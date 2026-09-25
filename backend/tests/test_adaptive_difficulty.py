from graph.nodes import adjust_difficulty
from graph.prompts import get_difficulty_guidance, get_interviewer_system_prompt


def test_strong_good_answer_increases_difficulty():
    assert adjust_difficulty("medium", "GOOD", "strong") == "hard"
    assert adjust_difficulty("easy", "GOOD", "strong") == "medium"
    assert adjust_difficulty("hard", "GOOD", "strong") == "hard"


def test_weak_or_vague_answer_decreases_difficulty():
    assert adjust_difficulty("hard", "VAGUE", "weak") == "medium"
    assert adjust_difficulty("medium", "GOOD", "weak") == "easy"
    assert adjust_difficulty("easy", "VAGUE", "weak") == "easy"


def test_adequate_good_answer_keeps_difficulty():
    assert adjust_difficulty("medium", "GOOD", "adequate") == "medium"


def test_prompts_include_difficulty_guidance():
    easy = get_interviewer_system_prompt("Software Engineer", stage="technical_fundamentals", difficulty="easy")
    hard = get_interviewer_system_prompt("Software Engineer", stage="technical_fundamentals", difficulty="hard")
    assert "EASY" in easy
    assert "HARD" in hard
    assert "accessible" in get_difficulty_guidance("easy").lower()
