"""
LangGraph State Schema for the Interview Practice Partner.
Defines the InterviewState TypedDict that flows through all graph nodes.
"""

from typing import TypedDict, Optional, List, Literal


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class FeedbackData(TypedDict):
    overallImpression: str
    communication: str
    technicalKnowledge: str
    strengths: str
    improvementAreas: List[str]


class InterviewState(TypedDict):
    """
    Central state object passed between all LangGraph nodes.

    Fields:
        session_id            — Unique session identifier
        role                  — The job role being interviewed for
        role_confirmed        — Whether the role has been extracted and confirmed

        -- Counters --
        main_question_count   — Number of main interview questions completed
        follow_up_count       — Number of follow-up probes asked (does NOT affect main count)
        is_follow_up          — True when the current exchange is a follow-up probe
        max_questions         — Target number of main questions (5-7, randomized)

        -- Interview progression (prevents repeated questions) --
        interview_stage       — Current stage: introduction/project_discussion/technical_fundamentals/system_design/behavioral
        previous_questions    — List of ALL main questions asked so far (verbatim)
        user_answers          — List of ALL user answers given so far (verbatim)

        -- Current turn --
        current_question      — The last question the agent asked
        history               — Full conversation history
        last_user_message     — Most recent user input
        classification        — Last classifier result
        classification_reason — One-line reason from classifier
        agent_response        — Text response to send to the frontend
        next_node             — Internal routing hint

        -- End state --
        feedback              — Structured feedback dict
        done                  — True when interview ended and feedback ready
        error                 — Optional error message
    """
    session_id: str
    role: Optional[str]
    role_confirmed: bool

    main_question_count: int
    follow_up_count: int
    is_follow_up: bool
    max_questions: int

    interview_stage: str
    previous_questions: List[str]
    user_answers: List[str]

    current_question: Optional[str]
    history: List[Message]
    last_user_message: str
    classification: Optional[Literal["VAGUE", "GOOD", "OFF_TOPIC", "OUT_OF_SCOPE"]]
    classification_reason: Optional[str]
    agent_response: Optional[str]
    next_node: Optional[str]
    project_question_count: int  # Tracks how many project-related questions have been asked (max 2)

    feedback: Optional[FeedbackData]
    done: bool
    error: Optional[str]
