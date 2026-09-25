"""
LangGraph node implementations for the Interview Practice Partner.

Each node is a pure function: (InterviewState) -> dict[str, Any]
The returned dict is merged into the state by LangGraph.

Node flow:
  START -> role_intake_node -> ask_question_node -> classify_answer_node
                                    ^   ^  ^  ^          |
                                    |   |  |  |          +- VAGUE -> follow_up_node --------+
                                    |   |  |  |          +- GOOD -> next_question_node      |
                                    |   |  |  |          +- OFF_TOPIC -> redirect_node -----+
                                    |   |  |  |          +- OUT_OF_SCOPE -> decline_node ---+
                                    +---+  +--+--------------------------------------------+
                                           (next_question_node -> END via generate_feedback_node)

Interview stages (7 questions total):
  introduction (Q1) -> project_discussion (Q2-Q3) -> technical_fundamentals (Q4-Q5) -> system_design (Q6) -> behavioral (Q7) -> feedback
"""

import logging
import random
import re
from typing import Any, Dict, Optional

from rag.retriever import ResumeRetriever
from .state import InterviewState
from .llm import chat_completion, structured_completion
from .prompts import (
    ROLE_INTAKE_SYSTEM_PROMPT,
    CLASSIFIER_SYSTEM_PROMPT,
    FEEDBACK_SYSTEM_PROMPT,
    FOLLOW_UP_SYSTEM_PROMPT,
    REDIRECT_SYSTEM_PROMPT,
    DECLINE_SYSTEM_PROMPT,
    ROLE_VALIDATOR_PROMPT,
    get_interviewer_system_prompt,
    get_difficulty_guidance,
    _get_role_topics,
)

logger = logging.getLogger(__name__)
resume_retriever = ResumeRetriever()

# -- Stop signal detection -------------------------------------------------
# IMPORTANT: We must NOT use substring matching. Words like "done", "finish",
# "quit", "stop" appear naturally in interview answers.

STOP_EXACT = {
    "stop", "done", "i'm done", "im done", "i am done",
    "finish", "quit", "end", "that's enough", "thats enough",
    "that is enough", "end interview", "end session",
    "no more questions", "end the interview",
    "i'm done with the interview", "i want to stop",
    "generate my feedback", "generate feedback",
    "please stop", "please end the interview",
}

STOP_PHRASES = {
    "end the interview", "end interview", "stop the interview",
    "no more questions", "generate my feedback", "generate feedback",
    "done with the interview", "i want to end the interview",
}


def _contains_stop_signal(text: str) -> bool:
    """Check if the user's message is an explicit stop/done signal."""
    lower = text.lower().strip()
    if lower in STOP_EXACT:
        return True
    return any(phrase in lower for phrase in STOP_PHRASES)


# -- Interview stage calculation -------------------------------------------
def _compute_stage(main_question_count: int, max_questions: int) -> str:
    """Determine the interview stage based on how many main questions
    have been completed.

    Stage mapping (for a 7-question interview):
      Q1 (index 0):        introduction
      Q2-Q3 (indices 1-2): project_discussion
      Q4-Q5 (indices 3-4): technical_fundamentals
      Q6 (index 5):        system_design
      Q7 (index 6):        behavioral
    """
    if main_question_count == 0:
        return "introduction"
    elif main_question_count in (1, 2):
        return "project_discussion"
    elif main_question_count in (3, 4):
        return "technical_fundamentals"
    elif main_question_count == 5:
        return "system_design"
    elif main_question_count == 6:
        return "behavioral"
    else:
        return "behavioral"


# -- Role extraction -------------------------------------------------------
def _extract_role_from_text(text: str) -> str | None:
    """Simple heuristic to extract a job role from free text."""
    common_roles = [
        "software engineer", "software developer", "frontend developer", "backend developer",
        "full stack developer", "data scientist", "data analyst", "data engineer",
        "machine learning engineer", "product manager", "product designer", "ux designer",
        "ui designer", "devops engineer", "cloud engineer", "site reliability engineer",
        "marketing manager", "sales representative", "sales manager", "account manager",
        "customer service", "customer support", "project manager", "scrum master",
        "business analyst", "financial analyst", "hr manager", "recruiter",
        "teacher", "professor", "nurse", "doctor", "lawyer", "accountant",
        "graphic designer", "content writer", "copywriter", "social media manager",
        "retail associate", "store manager", "operations manager",
    ]
    lower = text.lower()
    for role in common_roles:
        if role in lower:
            return role.title()

    patterns = [
        r"(?:for(?:\s+a(?:n)?)?|as(?:\s+a(?:n)?)?|apply(?:ing)?\s+for)\s+([a-zA-Z\s]{3,40}?)(?:\s+position|\s+role|\s+job|[,.\?!]|$)",
        r"(?:i(?:'m|\s+am)\s+a(?:n)?)\s+([a-zA-Z\s]{3,30}?)(?:[,.\?!]|$)",
        r"(?:interview\s+for)\s+([a-zA-Z\s]{3,40}?)(?:\s+position|\s+role|\s+job|[,.\?!]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            candidate = match.group(1).strip()
            if 2 < len(candidate) < 50:
                return candidate.title()
    return None


DIFFICULTY_LEVELS = ("easy", "medium", "hard")


def adjust_difficulty(current: str, classification: str, strength: Optional[str] = None) -> str:
    """Move interview difficulty up after strong answers and down after weak ones."""
    level = current if current in DIFFICULTY_LEVELS else "medium"
    index = DIFFICULTY_LEVELS.index(level)
    normalized_strength = (strength or "").lower()

    if classification == "GOOD" and normalized_strength == "strong":
        return DIFFICULTY_LEVELS[min(index + 1, len(DIFFICULTY_LEVELS) - 1)]
    if classification == "VAGUE" or normalized_strength == "weak":
        return DIFFICULTY_LEVELS[max(index - 1, 0)]
    return level


def _infer_strength(classification: str, provided: Optional[str], answer: str) -> str:
    valid = {"weak", "adequate", "strong"}
    if provided and provided.lower() in valid:
        return provided.lower()
    if classification == "VAGUE":
        return "weak"
    if classification in {"OFF_TOPIC", "OUT_OF_SCOPE"}:
        return "weak"
    if classification == "GOOD" and len(answer.strip()) > 280:
        return "strong"
    return "adequate"


# ---------------------------------------------------------------------------
# Role consistency validator
# ---------------------------------------------------------------------------
def _validate_question_role_alignment(question: str, role: str) -> bool:
    """
    Uses a lightweight LLM call to verify the generated question is truly
    relevant to the selected role.

    Returns True if aligned, False if the question has drifted to another domain.
    Falls back to True on LLM error (fail-open, never block the interview).
    """
    result = structured_completion(
        system_prompt=ROLE_VALIDATOR_PROMPT,
        user_content=f"Role: {role}\n\nGenerated question: {question}",
        temperature=0.0,
        max_tokens=80,
    )
    if result and isinstance(result, dict):
        aligned = bool(result.get("aligned", True))
        reason = result.get("reason", "")
        status = "PASS" if aligned else "FAIL"
        logger.info(
            "[ROLE CONSISTENCY CHECK] %s | Role=%s | Reason=%s | Q='%s'",
            status, role, reason, question[:80],
        )
        return aligned
    # Fail-open: if validator errors, allow the question through
    logger.warning("[ROLE CONSISTENCY CHECK] Validator returned no result — defaulting to PASS")
    return True


def role_intake_node(state: InterviewState) -> Dict[str, Any]:
    """
    Extracts the target job role from the user's message.
    If unclear, asks a clarifying question (loops back to itself).
    Sets role_confirmed=True once a role is identified.

    IMPORTANT: This node must NEVER be called after role_confirmed=True.
    The run_turn stage guard prevents this, but this hard guard is a
    second line of defence in case state is ever corrupted.
    """
    # Hard guard — should NEVER be triggered if run_turn is working correctly
    if state.get("role_confirmed"):
        logger.error(
            "[HARD GUARD] role_intake_node called while role_confirmed=True! "
            "Role='%s', Stage='%s'. Redirecting to classify_answer_node immediately.",
            state.get("role"), state.get("interview_stage"),
        )
        return {"next_node": "classify_answer_node"}

    user_msg = state["last_user_message"]
    history = state.get("history", [])

    extracted_role = _extract_role_from_text(user_msg)


    if extracted_role:
        response = (
            f"Great choice! Let's begin your **{extracted_role}** interview. "
            f"I'll ask you 5-7 questions covering behavioral and role-specific topics. "
            f"Take your time with each answer -- there's no rush. Ready? Let's go!"
        )
        new_history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
        logger.info(">>> role_intake_node: role='%s' confirmed -> INTERVIEW_ACTIVE", extracted_role)
        return {
            "role": extracted_role,
            "role_confirmed": True,
            "interview_stage": "INTERVIEW_ACTIVE",
            "agent_response": response,
            "history": new_history,
            "question_generation_failed": False,
            "ready_for_feedback": False,
            "next_node": "ask_question_node",
        }
    else:
        response = chat_completion(
            system_prompt=ROLE_INTAKE_SYSTEM_PROMPT,
            history=history,
            user_message=user_msg,
            temperature=0.5,
            max_tokens=200,
        )
        if response is None:
            response = (
                "I'm temporarily unable to continue right now due to API traffic. "
                "Please wait a moment and try again."
            )
            new_history = history + [
                {"role": "assistant", "content": response},
            ]
            return {
                "role": None,
                "role_confirmed": False,
                "agent_response": response,
                "history": new_history,
                "next_node": "role_intake_node",
            }

        new_history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
        logger.info(">>> role_intake_node: role unclear, looping")
        return {
            "role": None,
            "role_confirmed": False,
            "agent_response": response,
            "history": new_history,
            "next_node": "role_intake_node",
        }


# ---------------------------------------------------------------------------
# Node 2: ask_question_node
# ---------------------------------------------------------------------------
def ask_question_node(state: InterviewState) -> Dict[str, Any]:
    """
    Generates the next interview question.

    Key features:
    - Stage-aware: introduction → project_discussion → technical_fundamentals → system_design → behavioral
    - Duplicate prevention: passes ALL previously asked questions to the LLM
    - Role consistency: validates every question against the selected role;
      retries with a stricter prompt if the question drifts off-role
    """
    role = state.get("role", "General")
    history = state.get("history", [])
    session_id = state.get("session_id")
    main_question_count = state.get("main_question_count", 0)
    max_questions = state.get("max_questions", 6)
    previous_questions = state.get("previous_questions", [])
    resume_chunks = []
    if session_id and state.get("resume_uploaded"):
        resume_chunks = resume_retriever.retrieve_context(
            session_id=session_id,
            role=role,
            history=history,
            top_k=5,
        )
    user_answers = state.get("user_answers", [])

    # Compute the current interview stage
    stage = _compute_stage(main_question_count, max_questions)
    project_question_count = state.get("project_question_count", 0)
    difficulty = state.get("difficulty", "medium")
    system_prompt = get_interviewer_system_prompt(role, stage=stage, difficulty=difficulty)

    # Build stage-specific instruction
    if main_question_count == 0:
        instruction = (
            f"This is the FIRST question of the {role} interview. "
            f"Stage: {stage.upper()}. "
            f"Briefly introduce yourself as the interviewer, then ask a warm-up question "
            f"asking the candidate to introduce themselves and mention the most relevant internship, project, "
            f"or certification from their resume for the {role} role."
        )
    elif stage == "project_discussion":
        instruction = (
            f"This is question #{main_question_count + 1} of {max_questions}. "
            f"Stage: PROJECT DISCUSSION (asking about the candidate's resume experiences). "
            f"Ask a focused question about one of the candidate's resume projects, internships, or certifications. "
            f"Focus on technical decisions, challenges faced, measurable outcomes, and lessons learned."
        )
    elif stage == "technical_fundamentals":
        instruction = (
            f"This is question #{main_question_count + 1} of {max_questions}. "
            f"Stage: TECHNICAL FUNDAMENTALS. "
            f"Ask a core {role} technical question that tests fundamental knowledge. "
            f"Topics: algorithms, data structures, design patterns, APIs, databases, etc. "
            f"Do NOT ask about the candidate's personal projects. "
            f"Ask about general {role} technical concepts."
        )
    elif stage == "system_design":
        instruction = (
            f"This is question #{main_question_count + 1} of {max_questions}. "
            f"Stage: SYSTEM DESIGN / SCALABILITY. "
            f"Ask a system design or scalability question relevant to {role}. "
            f"Examples: design a URL shortener, caching strategies, handling scale, etc. "
            f"Do NOT ask about personal projects. Focus on architectural thinking."
        )
    elif stage == "behavioral":
        instruction = (
            f"This is question #{main_question_count + 1} of {max_questions}. "
            f"Stage: BEHAVIORAL / SOFT SKILLS. "
            f"Ask a behavioral question about teamwork, communication, conflict resolution, or deadline management. "
            f"Do NOT ask about technical topics or projects. "
            f"Use the STAR format (Situation, Task, Action, Result)."
        )
    else:
        instruction = (
            f"This is question #{main_question_count + 1} of {max_questions}. "
            f"Stage: {stage.upper()}. "
            f"Ask the next {role} interview question."
        )

    instruction += f"\n\n{get_difficulty_guidance(difficulty)}"

    # CRITICAL: Tell the LLM exactly which questions were already asked
    if previous_questions:
        questions_list = "\n".join(
            f"  {i+1}. {q[:120]}" for i, q in enumerate(previous_questions)
        )
        instruction += (
            f"\n\nALREADY ASKED ({len(previous_questions)} questions) — "
            f"do NOT repeat or rephrase any of these:\n{questions_list}"
        )

    # Reference the user's last answer for natural contextual progression,
    # BUT explicitly remind the LLM to stay on role
    if user_answers and main_question_count > 0:
        last_answer = user_answers[-1][:150]
        instruction += (
            f"\n\nCandidate's last answer (use for context ONLY — "
            f"your question must still be about {role} skills): \"{last_answer}\""
        )

    if resume_chunks:
        context_block = "\n".join(f"- {chunk}" for chunk in resume_chunks)
        instruction += (
            f"\n\nResume Context:\n{context_block}\n\n"
            f"Priority: ask about the candidate's resume projects, internships, or certifications when possible. "
            f"If the resume contains relevant details, reference them in the question or use them to narrow the focus."
        )

    # Generate question, then validate role alignment. Retry once with a
    # stricter prompt if the first attempt fails the consistency check.
    MAX_RETRIES = 2
    question = None
    for attempt in range(MAX_RETRIES):
        candidate_q = chat_completion(
            system_prompt=system_prompt,
            history=history,
            user_message=instruction,
            temperature=0.8 if attempt == 0 else 0.5,  # lower temp on retry
            max_tokens=300,
        )

        if candidate_q is None:
            error_msg = (
                "I'm temporarily unable to generate the next question due to API traffic. "
                "Please wait a moment and try again."
            )
            new_history = history + [{"role": "assistant", "content": error_msg}]
            return {
                "agent_response": error_msg,
                "history": new_history,
                "question_generation_failed": True,
                "next_node": "ask_question_node",
            }

        if _validate_question_role_alignment(candidate_q, role):
            question = candidate_q
            break
        else:
            logger.warning(
                "[ROLE DRIFT] attempt=%d question failed consistency check. "
                "Retrying with stricter prompt. Role=%s, Q='%s'",
                attempt + 1, role, candidate_q[:80],
            )
            # Tighten the instruction for retry
            topics = _get_role_topics(role)
            instruction = (
                f"STRICT RETRY: Generate a {role} interview question for stage {stage.upper()}. "
                f"The question MUST be directly about one of these topics: "
                f"{', '.join(topics[:10])}. "
                f"Do NOT reference any personal projects in the question itself. "
                f"Just ask a focused {role} interview question."
            )
            if previous_questions:
                instruction += (
                    f" Do NOT repeat any of these already-asked questions: "
                    f"{'; '.join(q[:60] for q in previous_questions[-3:])}"
                )

    # If all retries failed, use last attempt anyway (never block the interview)
    if question is None:
        question = candidate_q
        logger.warning("[ROLE CONSISTENCY] All retries failed. Using last attempt anyway.")

    # Store in previous_questions for duplicate prevention
    updated_previous = previous_questions + [question]
    new_history = history + [{"role": "assistant", "content": question}]

    # Track project questions (only count during project_discussion stage)
    updated_project_count = project_question_count
    if stage == "project_discussion":
        updated_project_count += 1

    logger.info(
        "[ASK] Stage=%s | Role=%s | Difficulty=%s | Q=%d/%d | project_q=%d | prev_count=%d | q='%s'",
        stage, role, difficulty, main_question_count, max_questions,
        updated_project_count, len(updated_previous), question[:80],
    )

    return {
        "current_question": question,
        "agent_response": question,
        "history": new_history,
        "is_follow_up": False,
        "interview_stage": stage,
        "previous_questions": updated_previous,
        "project_question_count": updated_project_count,
        "question_generation_failed": False,
        "next_node": "classify_answer_node",
    }


# ---------------------------------------------------------------------------
# Node 3: classify_answer_node (router)
# ---------------------------------------------------------------------------
def classify_answer_node(state: InterviewState) -> Dict[str, Any]:
    """
    Classifies the user's answer and routes to the appropriate handler.

    IMPORTANT: This node adds the user's turn to history. All downstream
    nodes receive history that already includes the user message.
    Also stores the user's answer in user_answers for context tracking.
    """
    user_msg = state["last_user_message"]
    current_question = state.get("current_question", "the previous question")
    history = state.get("history", [])
    user_answers = state.get("user_answers", [])

    # Add user message to history and track the answer
    updated_history = history + [{"role": "user", "content": user_msg}]
    updated_answers = user_answers + [user_msg]

    # Check for explicit stop signal before classifying
    if _contains_stop_signal(user_msg):
        logger.info(">>> classify_answer_node: STOP signal detected in '%s'", user_msg[:60])
        return {
            "classification": "GOOD",
            "classification_reason": "User requested to end the interview",
            "answer_strength": "adequate",
            "history": updated_history,
            "user_answers": updated_answers,
            "next_node": "next_question_node",
        }

    prompt_content = (
        f"Interview Question: {current_question}\n\n"
        f"Candidate's Answer: {user_msg}"
    )

    result = structured_completion(
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        user_content=prompt_content,
        temperature=0.1,
        max_tokens=100,
    )

    valid_classifications = {"VAGUE", "GOOD", "OFF_TOPIC", "OUT_OF_SCOPE"}

    if result and isinstance(result, dict):
        classification = result.get("classification", "").upper()
        reason = result.get("reason", "")
        if classification in valid_classifications:
            strength = _infer_strength(classification, result.get("strength"), user_msg)
            difficulty = adjust_difficulty(
                state.get("difficulty", "medium"),
                classification,
                strength,
            )
            strong_streak = state.get("consecutive_strong_answers", 0)
            weak_streak = state.get("consecutive_weak_answers", 0)
            if strength == "strong" and classification == "GOOD":
                strong_streak += 1
                weak_streak = 0
            elif classification == "VAGUE" or strength == "weak":
                weak_streak += 1
                strong_streak = 0
            else:
                strong_streak = 0
                weak_streak = 0

            logger.info(
                ">>> classify_answer_node: %s/%s -- %s (stage=%s, main_q=%d, difficulty=%s, is_follow_up=%s)",
                classification, strength, reason,
                state.get("interview_stage", "?"),
                state.get("main_question_count", 0),
                difficulty,
                state.get("is_follow_up", False),
            )
            extra = {
                "answer_strength": strength,
                "difficulty": difficulty,
                "consecutive_strong_answers": strong_streak,
                "consecutive_weak_answers": weak_streak,
            }
            if classification in ("OFF_TOPIC", "OUT_OF_SCOPE"):
                extra["is_follow_up"] = False
            return {
                "classification": classification,
                "classification_reason": reason,
                "history": updated_history,
                "user_answers": updated_answers,
                "next_node": _classification_to_node(classification),
                **extra,
            }

    # Fail open: ask for more detail if the classifier couldn't respond.
    logger.warning(
        ">>> classify_answer_node: classifier unavailable or malformed output; asking for more detail. Raw: %r",
        result,
    )
    return {
        "classification": "VAGUE",
        "classification_reason": (
            "Could not classify the answer due to a temporary system issue; "
            "asking for more detail."
        ),
        "history": updated_history,
        "user_answers": updated_answers,
        "next_node": "follow_up_node",
    }


def _classification_to_node(classification: str) -> str:
    return {
        "VAGUE": "follow_up_node",
        "GOOD": "next_question_node",
        "OFF_TOPIC": "redirect_node",
        "OUT_OF_SCOPE": "decline_node",
    }.get(classification, "next_question_node")


# ---------------------------------------------------------------------------
# Node 4: follow_up_node (triggered on VAGUE)
# ---------------------------------------------------------------------------
def follow_up_node(state: InterviewState) -> Dict[str, Any]:
    """
    Asks a dynamic follow-up question based on the user's actual answer.
    Sets is_follow_up=True so next_question_node won't increment the counter.
    """
    history = state.get("history", [])
    user_msg = state["last_user_message"]
    current_question = state.get("current_question", "your last answer")
    follow_up_count = state.get("follow_up_count", 0) + 1
    session_id = state.get("session_id")
    role = state.get("role", "General")
    resume_chunks = []
    if session_id and state.get("resume_uploaded"):
        resume_chunks = resume_retriever.retrieve_context(
            session_id=session_id,
            role=role,
            history=history,
            top_k=3,
        )

    instruction = (
        f"The candidate just gave this vague answer to the question '{current_question}': "
        f"'{user_msg}'. Ask a specific follow-up question to probe for more detail. "
        f"Reference what they said and ask for a concrete example or measurable outcome."
    )

    if resume_chunks:
        context_block = "\n".join(f"- {chunk}" for chunk in resume_chunks)
        instruction += (
            f"\n\nResume Context:\n{context_block}\n\n"
            f"If possible, make the follow-up question connect to the candidate's resume experience, "
            f"especially their projects, internships, or certifications."
        )

    follow_up = chat_completion(
        system_prompt=FOLLOW_UP_SYSTEM_PROMPT,
        history=history,
        user_message=instruction,
        temperature=0.6,
        max_tokens=150,
    )

    if follow_up is None:
        follow_up = (
            "I'm temporarily unable to continue right now due to API traffic. "
            "Please wait a moment and try again."
        )
        new_history = history + [{"role": "assistant", "content": follow_up}]
        return {
            "agent_response": follow_up,
            "history": new_history,
            "is_follow_up": True,
            "follow_up_count": follow_up_count,
            "next_node": "follow_up_node",
        }

    logger.info(
        ">>> follow_up_node: follow_up_count=%d, stage=%s, probing='%s'",
        follow_up_count, state.get("interview_stage", "?"), follow_up[:80]
    )

    new_history = history + [{"role": "assistant", "content": follow_up}]
    return {
        "agent_response": follow_up,
        "history": new_history,
        "is_follow_up": True,
        "follow_up_count": follow_up_count,
        "next_node": "classify_answer_node",
    }


# ---------------------------------------------------------------------------
# Node 5: next_question_node (triggered on GOOD)
# ---------------------------------------------------------------------------
def next_question_node(state: InterviewState) -> Dict[str, Any]:
    """
    Decides whether to ask another question or end the interview.

    Increments main_question_count ONLY when a main question was completed
    (not when a follow-up answer was classified as GOOD).

    Termination fires ONLY when:
      - main_question_count >= max_questions (5-7), OR
      - User explicitly requested to stop
    """
    user_msg = state["last_user_message"]
    is_follow_up = state.get("is_follow_up", False)
    current_count = state.get("main_question_count", 0)
    max_questions = state.get("max_questions", 6)
    history = state.get("history", [])

    # Entry log — makes the is_follow_up / counter state visible on every turn
    logger.info(
        "[next_question_node ENTRY] incoming is_follow_up=%s, current_count=%d, "
        "classification=%s",
        is_follow_up, current_count,
        state.get("classification", "N/A"),
    )

    # Always increment: next_question_node is only reached on a GOOD classification.
    # Whether the GOOD answer was the first attempt OR resolved a follow-up, the
    # current main question is now complete either way.
    main_question_count = current_count + 1

    user_wants_stop = _contains_stop_signal(user_msg)

    # Enforce a minimum of 3 answered questions before allowing stop-signal termination.
    # This ensures feedback has enough transcript to be meaningful.
    MIN_QUESTIONS_FOR_STOP = 3
    early_stop = user_wants_stop and main_question_count < MIN_QUESTIONS_FOR_STOP

    interview_complete = (main_question_count >= max_questions) or (
        user_wants_stop and not early_stop
    )

    # Compute the next stage
    next_stage = _compute_stage(main_question_count, max_questions)

    logger.info(
        "[next_question_node] main_q=%d/%d (was %d), is_follow_up=%s, "
        "stage=%s, user_wants_stop=%s, early_stop=%s, complete=%s",
        main_question_count, max_questions, current_count, is_follow_up,
        next_stage, user_wants_stop, early_stop, interview_complete,
    )

    if early_stop:
        # User wants to stop but we don't have enough transcript for good feedback yet
        remaining = MIN_QUESTIONS_FOR_STOP - main_question_count
        nudge_msg = (
            f"I appreciate that — we're almost there! "
            f"Let's do just {remaining} more question{'s' if remaining > 1 else ''} "
            f"so I can give you truly useful, specific feedback. Ready?"
        )
        new_history = history + [{"role": "assistant", "content": nudge_msg}]
        logger.info(
            "[next_question_node] Early stop nudge: main_q=%d, need %d more",
            main_question_count, remaining,
        )
        return {
            "main_question_count": main_question_count,
            "is_follow_up": False,
            "interview_stage": next_stage,
            "history": new_history,
            "agent_response": nudge_msg,
            "next_node": "ask_question_node",
        }

    elif interview_complete:
        if user_wants_stop:
            transition_msg = (
                "Of course! Let's wrap up here. "
                "I'll now prepare your personalized feedback -- one moment..."
            )
        else:
            transition_msg = (
                f"That completes our interview! I asked you {main_question_count} questions "
                f"and I appreciate the thoughtful answers. "
                f"Let me now prepare your detailed feedback report..."
            )
        new_history = history + [{"role": "assistant", "content": transition_msg}]
        return {
            "main_question_count": main_question_count,
            "is_follow_up": False,
            "interview_stage": "feedback",
            "history": new_history,
            "agent_response": transition_msg,
            "next_node": "generate_feedback_node",
        }
    else:
        # Brief contextual acknowledgment before moving on
        acknowledgments = [
            "Thanks for sharing that.",
            "Good, I appreciate that answer.",
            "Interesting perspective, thank you.",
            "Got it, thank you.",
            "That's helpful context.",
        ]
        ack = random.choice(acknowledgments)
        new_history = history + [{"role": "assistant", "content": ack}]
        return {
            "main_question_count": main_question_count,
            "is_follow_up": False,
            "interview_stage": next_stage,
            "history": new_history,
            "agent_response": ack,
            "next_node": "ask_question_node",
        }


# ---------------------------------------------------------------------------
# Node 6: redirect_node (triggered on OFF_TOPIC)
# ---------------------------------------------------------------------------
def redirect_node(state: InterviewState) -> Dict[str, Any]:
    """
    Briefly acknowledges off-topic answer and redirects back to current question.
    """
    history = state.get("history", [])
    user_msg = state["last_user_message"]
    current_question = state.get("current_question", "my previous question")

    instruction = (
        f"The candidate went off-topic. Their answer was: '{user_msg}'. "
        f"Politely redirect them back to: '{current_question}'"
    )

    redirect_response = chat_completion(
        system_prompt=REDIRECT_SYSTEM_PROMPT,
        history=history,
        user_message=instruction,
        temperature=0.5,
        max_tokens=100,
    )

    if redirect_response is None:
        redirect_response = (
            "I’m temporarily unable to continue right now due to API traffic. "
            "Please wait a moment and try again."
        )
        new_history = history + [{"role": "assistant", "content": redirect_response}]
        return {
            "agent_response": redirect_response,
            "history": new_history,
            "next_node": "redirect_node",
        }

    logger.info(">>> redirect_node: OFF_TOPIC, redirecting to current question")

    new_history = history + [{"role": "assistant", "content": redirect_response}]
    return {
        "agent_response": redirect_response,
        "history": new_history,
        "next_node": "classify_answer_node",
    }


# ---------------------------------------------------------------------------
# Node 7: decline_node (triggered on OUT_OF_SCOPE)
# ---------------------------------------------------------------------------
def decline_node(state: InterviewState) -> Dict[str, Any]:
    """
    Politely declines out-of-scope requests, stays in persona, redirects.
    """
    history = state.get("history", [])
    user_msg = state["last_user_message"]
    current_question = state.get("current_question", "the current interview question")

    instruction = (
        f"The candidate made an out-of-scope request: '{user_msg}'. "
        f"Politely decline and redirect them back to: '{current_question}'"
    )

    decline_response = chat_completion(
        system_prompt=DECLINE_SYSTEM_PROMPT,
        history=history,
        user_message=instruction,
        temperature=0.4,
        max_tokens=120,
    )

    if decline_response is None:
        decline_response = (
            "I’m temporarily unable to continue right now due to API traffic. "
            "Please wait a moment and try again."
        )
        new_history = history + [{"role": "assistant", "content": decline_response}]
        return {
            "agent_response": decline_response,
            "history": new_history,
            "next_node": "decline_node",
        }

    logger.info(">>> decline_node: OUT_OF_SCOPE, declining and redirecting")

    new_history = history + [{"role": "assistant", "content": decline_response}]
    return {
        "agent_response": decline_response,
        "history": new_history,
        "next_node": "classify_answer_node",
    }


# ---------------------------------------------------------------------------
# Node 8: generate_feedback_node
# ---------------------------------------------------------------------------
def generate_feedback_node(state: InterviewState) -> Dict[str, Any]:
    """
    Separate LLM call that analyzes the full transcript and returns
    structured JSON feedback with specific examples from the user's answers.
    The feedback is explicitly labelled with the selected role so it is
    never generic.
    """
    history = state.get("history", [])
    role = state.get("role", "General")
    session_id = state.get("session_id")
    main_question_count = state.get("main_question_count", 0)

    # Build readable transcript from history
    transcript_lines = []
    for msg in history:
        prefix = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        transcript_lines.append(f"{prefix}: {msg['content']}")
    transcript = "\n\n".join(transcript_lines)

    resume_context = []
    if session_id and state.get("resume_uploaded"):
        resume_context = resume_retriever.retrieve_context(
            session_id=session_id,
            role=role,
            history=history,
            top_k=5,
        )

    prompt_content = (
        f"INTERVIEW ROLE: {role}\n"
        f"TOTAL MAIN QUESTIONS ANSWERED: {main_question_count}\n\n"
        f"IMPORTANT: All feedback must be specific to the {role} role. "
        f"Reference the actual role when evaluating technical knowledge and skills. "
        f"Title your overallImpression as '{role} Interview Feedback'.\n\n"
        f"Full Interview Transcript:\n{transcript}\n\n"
    )
    if resume_context:
        prompt_content += (
            "Resume Context:\n" + "\n".join(f"- {chunk}" for chunk in resume_context) + "\n\n"
        )
    prompt_content += "Please provide structured feedback on this interview performance."

    logger.info(
        "[FEEDBACK] Generating %s Interview Feedback | Q=%d | transcript_len=%d chars",
        role, main_question_count, len(transcript),
    )

    result = structured_completion(
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
        user_content=prompt_content,
        temperature=0.3,
        max_tokens=800,
    )

    if result and isinstance(result, dict):
        # Clamp readinessScore to 1-10 range if provided
        raw_score = result.get("readinessScore")
        try:
            score = max(1, min(10, int(raw_score))) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None
        feedback = {
            "interviewRole": role,
            "readinessScore": score,
            "overallImpression": result.get("overallImpression", "Feedback unavailable."),
            "communication": result.get("communication", "Feedback unavailable."),
            "technicalKnowledge": result.get("technicalKnowledge", "Feedback unavailable."),
            "strengths": result.get("strengths", "Feedback unavailable."),
            "improvementAreas": result.get("improvementAreas", ["Keep practicing!"]),
        }
    else:
        logger.warning("[FEEDBACK] Failed to get structured feedback. Using fallback.")
        feedback = {
            "interviewRole": role,
            "readinessScore": None,
            "overallImpression": (
                f"{role} Interview Feedback: You completed the session. Well done for practicing!"
            ),
            "communication": "Unable to analyze communication patterns at this time.",
            "technicalKnowledge": (
                f"Unable to evaluate {role}-specific technical responses at this time."
            ),
            "strengths": "Your willingness to practice shows commitment to improvement.",
            "improvementAreas": [
                "Consider using the STAR method (Situation, Task, Action, Result) for behavioral questions.",
                "Practice giving more specific examples with measurable outcomes.",
            ],
        }

    logger.info("[FEEDBACK] Done -> INTERVIEW_COMPLETE | Role=%s", role)

    done_message = (
        f"Your {role} interview session is complete! "
        "Here's your personalized feedback report."
    )
    return {
        "feedback": feedback,
        "done": True,
        "interview_stage": "INTERVIEW_COMPLETE",
        "agent_response": done_message,
        "next_node": "__end__",
    }
