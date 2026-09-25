# Interview Practice Partner

An AI-powered mock interview coach that conducts adaptive, role-specific interviews via voice or text and delivers structured performance feedback. Built with a LangGraph agent graph, a Groq LLM backend, and a vanilla-JS frontend with browser-native speech I/O — no external STT/TTS services required.

---

## Architecture

The backend is a compiled LangGraph `StateGraph`. Flask persists `InterviewState` as JSON in SQLite and resumes the graph checkpoint on each `/chat` turn. Strong answers raise difficulty; weak or vague answers trigger a clarification and lower difficulty.

```
START
  │
  ▼
role_intake_node          ← Extracts / confirms the target role from user input.
                            Sets role_confirmed=True. NEVER called again after this.
  │
  ▼
ask_question_node         ← Generates the next interview question.
                            Stage-aware (intro → behavioral → technical → advanced).
                            Validates every question against the selected role;
                            retries with a stricter prompt if role drift is detected.
  │
  ▼
classify_answer_node      ← Classifies the user's answer:
  │                         GOOD / VAGUE / OFF_TOPIC / OUT_OF_SCOPE
  ├─ GOOD ──────────────▶ next_question_node
  │                         Increments main_question_count (always, even after follow-up).
  │                         Enforces a minimum of 3 answered questions before stop signals
  │                         are honoured. Routes to generate_feedback_node when complete.
  │
  ├─ VAGUE ─────────────▶ follow_up_node
  │                         Asks a focused follow-up question. Sets is_follow_up=True.
  │                         Routes back to classify_answer_node.
  │
  ├─ OFF_TOPIC ─────────▶ redirect_node
  │                         Politely redirects the candidate back to the question.
  │
  └─ OUT_OF_SCOPE ──────▶ decline_node
                            Stays in character, declines inappropriate requests,
                            redirects back to the interview.

generate_feedback_node    ← Analyses the full transcript with a separate LLM call.
                            Returns structured JSON: readinessScore (1-10),
                            overallImpression, communication, technicalKnowledge,
                            strengths, improvementAreas.
```

**Stage-aware guards:** once `role_confirmed` is true, the compiled graph enters at `classify_answer_node` instead of role intake.

---

## Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (model: `claude-sonnet-4-5` or equivalent)
- Chrome/Edge for voice input (browser Web Speech API)

### 1. Clone and create the virtual environment
```bash
git clone https://github.com/YOUR_USERNAME/interview_practice_partner.git
cd interview_practice_partner
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_...
```

### 4. Run the backend
```bash
cd backend
python app.py
# Backend runs on http://localhost:5000
```

### 5. Open the frontend
Open `frontend/index.html` in Chrome or Edge (double-click, or serve with any static file server).

---

## Design Decisions

### Why a node graph instead of a single prompt?
A single mega-prompt that must handle role intake, question generation, answer classification, follow-ups, redirects, and feedback all at once becomes brittle and hard to debug. The node graph makes each responsibility explicit and independently testable. When a bug appears (e.g., role re-detection mid-interview), the exact node responsible is immediately obvious from the logs.

### Why a compiled LangGraph?
The interview is interactive — it pauses between turns for user input. The compiled graph uses `interrupt_before` on the nodes that need the next candidate message, then Flask resumes the same SQLite checkpoint. Routing is done with conditional edges (GOOD / VAGUE / OFF_TOPIC / OUT_OF_SCOPE), not a Flask if/else dispatcher.

### Why Groq?
Groq's inference speed is significantly faster than most providers, which matters for a real-time interview UX. The free tier is reliable for development and demos. The `groq` Python SDK is also straightforward — no LangChain wrapper needed.

### Why browser-native voice APIs?
The Web Speech API (SpeechRecognition + SpeechSynthesis) requires zero server infrastructure and zero API cost. For a demo-grade project it is the right trade-off: it works instantly in Chrome/Edge with a single permission prompt. The downside (Chrome-only, quality varies) is documented in Known Limitations.

### Why text input is always the primary input
Voice is a nice-to-have enhancement, not a requirement. The text input is always visible, always enabled, and is the fallback if the browser denies microphone access or if the user is in a noisy environment. This prevents the app from being unusable in incognito mode or on Firefox.

---

## Testing

From the repo root:

```bash
pip install -r backend/requirements.txt
pytest
```

Automated coverage includes role extraction, answer routing, stop-signal handling, stage transitions, SQLite session persistence, resume isolation, compiled-graph execution, and Flask API endpoints.

---

## Known Limitations

- **STT quality:** Browser Speech Recognition works best in Chrome on a desktop. Mobile Chrome is variable; Firefox is not supported.
- **Feedback depth:** Feedback quality scales with transcript length. Short sessions (< 3 questions) produce generic feedback — this is why the app enforces a minimum of 3 answered questions before stopping.
- **Role topic coverage:** The role-topic validator only covers ~10 common roles. Niche or unusual roles (e.g. "Radiological Technician") fall back to generic professional topics.
- **State is SQLite-backed:** interview JSON and LangGraph checkpoints survive backend restarts. For a larger production deployment, Redis or PostgreSQL can still replace the local files.
- **LLM non-determinism:** Even with role anchoring, occasional question drift is possible. The validator catches and retries most cases; persistent drift would require a more constrained few-shot prompt.
