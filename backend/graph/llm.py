"""
Groq API wrapper with retry logic, history truncation, and structured output support.

Uses the official `groq` SDK with tenacity for exponential-backoff retries on rate limits.
"""

import os
import json
import logging
import re
from typing import List, Optional
from groq import Groq, RateLimitError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------
_client: Optional[Groq] = None

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


# Groq model IDs can change over time; keep this configurable so deployments
# can select an account-supported model without changing application code.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Maximum number of history turns to send to avoid hitting token limits.
# Keeps the system prompt + last N user/assistant pairs.
# Reduced to 8 to lower token usage and avoid rate limits.
MAX_HISTORY_TURNS = 8  # 4 pairs = 8 messages


# ---------------------------------------------------------------------------
# History truncation
# ---------------------------------------------------------------------------
def truncate_history(messages: List[dict], max_turns: int = MAX_HISTORY_TURNS) -> List[dict]:
    """
    Keep the system prompt (always first) and the most recent `max_turns` messages.
    This prevents token overflow on long conversations.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    
    # Keep only the last max_turns non-system messages
    if len(non_system) > max_turns:
        non_system = non_system[-max_turns:]
    
    return system_msgs + non_system


# ---------------------------------------------------------------------------
# Retry decorator — retries on rate limit (429) with exponential back-off
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_groq(messages: List[dict], temperature: float = 0.7, max_tokens: int = 1024) -> Optional[str]:
    """Raw Groq call with retry logic. Returns the assistant content string."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content and content.strip() else None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(raw: Optional[str]) -> Optional[dict]:
    """Parse JSON from raw LLM output, tolerating markdown fences and prose.

    Returns the parsed dict, or None when no valid JSON object is found.
    """
    if not raw:
        return None
    text = raw.strip()

    # Prefer a fenced block if present (handles multi- and single-line fences).
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Direct parse first.
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to the first {...} span in the output.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def chat_completion(
    system_prompt: str,
    history: List[dict],
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Standard chat completion for question-asking and persona nodes.
    Automatically truncates history to control token usage.
    Returns the assistant response string, or None on failure.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})
    messages = truncate_history(messages)

    try:
        return _call_groq(messages, temperature=temperature, max_tokens=max_tokens)
    except RateLimitError:
        logger.error("Groq rate limit: max retries exhausted after 8 attempts.")
        return None
    except APIStatusError as e:
        logger.error("Groq API error: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected LLM error: %s", e)
        return None


def structured_completion(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> Optional[dict]:
    """
    Completion for nodes that require JSON structured output (classifier, feedback).
    Low temperature for deterministic classification. Strips markdown fences and
    retries the LLM call once when the first response is not valid JSON.
    Returns parsed dict or None on failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    def _attempt() -> Optional[dict]:
        try:
            raw = _call_groq(messages, temperature=temperature, max_tokens=max_tokens)
        except RateLimitError:
            logger.error("Groq rate limit in structured_completion.")
            return None
        except Exception as e:
            logger.error("structured_completion error: %s", e)
            return None
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("structured_completion: JSON parse failed. Raw: %r", raw)
        return parsed

    parsed = _attempt()
    if parsed is None:
        # One retry with an explicit nudge toward bare JSON.
        logger.info("structured_completion: retrying once after parse failure")
        messages = messages + [
            {
                "role": "user",
                "content": "Your last response was not valid JSON. Reply with ONLY the JSON object, no fences, no prose.",
            }
        ]
        parsed = _attempt()
    return parsed
