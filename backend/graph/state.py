"""
LangGraph State Schema for the Interview Practice Partner.
Defines the InterviewState TypedDict that flows through all graph nodes.
"""

from typing import List, Literal, Optional, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class FeedbackData(TypedDict):
    overallImpression: str
    communication: str
    technicalKnowledge: str
    strengths: str
    improvementAreas: List[str]


DifficultyLevel = Literal["easy", "medium", "hard"]
AnswerStrength = Literal["weak", "adequate", "strong"]


class InterviewState(TypedDict):
    """
    Central state object passed between all LangGraph nodes.
    """
    session_id: str
    role: Optional[str]
    role_confirmed: bool

    main_question_count: int
    follow_up_count: int
    follow_ups_this_question: int
    is_follow_up: bool
    max_questions: int

    interview_stage: str
    previous_questions: List[str]
    user_answers: List[str]

    current_question: Optional[str]
    history: List[Message]
    last_user_message: str
    classification: Optional[Literal["VAGUE", "GOOD", "OFF_TOPIC", "OUT_OF_SCOPE", "RESUME_QA"]]
    classification_reason: Optional[str]
    answer_strength: Optional[AnswerStrength]
    difficulty: DifficultyLevel
    consecutive_strong_answers: int
    consecutive_weak_answers: int
    agent_response: Optional[str]
    next_node: Optional[str]
    question_generation_failed: bool
    ready_for_feedback: bool
    project_question_count: int

    feedback: Optional[FeedbackData]
    done: bool
    error: Optional[str]
    resume_uploaded: bool
    resume_filename: Optional[str]
    resume_chunk_count: int


STAGE_ROLE_SELECTION = "ROLE_SELECTION"
STAGE_INTERVIEW_ACTIVE = "INTERVIEW_ACTIVE"
STAGE_FEEDBACK = "FEEDBACK_GENERATION"
STAGE_COMPLETE = "INTERVIEW_COMPLETE"


def create_initial_state(session_id: str) -> dict:
    """Create a fresh InterviewState for a new session."""
    return {
        "session_id": session_id,
        "role": None,
        "role_confirmed": False,
        "main_question_count": 0,
        "follow_up_count": 0,
        "follow_ups_this_question": 0,
        "is_follow_up": False,
        "max_questions": 7,
        "interview_stage": STAGE_ROLE_SELECTION,
        "previous_questions": [],
        "user_answers": [],
        "current_question": None,
        "history": [],
        "last_user_message": "",
        "classification": None,
        "classification_reason": None,
        "answer_strength": None,
        "difficulty": "medium",
        "consecutive_strong_answers": 0,
        "consecutive_weak_answers": 0,
        "agent_response": None,
        "next_node": "role_intake_node",
        "question_generation_failed": False,
        "ready_for_feedback": False,
        "project_question_count": 0,
        "feedback": None,
        "done": False,
        "error": None,
        "resume_uploaded": False,
        "resume_filename": None,
        "resume_chunk_count": 0,
    }
