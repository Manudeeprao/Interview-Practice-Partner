# Interview Practice Partner

An AI mock interview app that runs a role-based interview, adapts question difficulty based on answers, and uses a candidate resume to personalize follow-up questions.

## What it does

- interviews the user in a role-specific workflow (7 questions: intro, projects, technical, system design, behavioral)
- stores session state in SQLite between requests
- resumes the same compiled LangGraph graph for each session
- classifies answers as GOOD, VAGUE, OFF_TOPIC, or OUT_OF_SCOPE
- raises or lowers difficulty depending on answer quality
- caps follow-ups at 2 per question so the interview can never stall
- accepts resume uploads in PDF and DOCX format, with section-aware chunking and per-session ChromaDB retrieval
- grounds questions and feedback in the uploaded resume (never invents resume details)
- generates end-of-interview feedback with a 1–10 readiness score
- supports browser microphone input (speech recognition) and text-to-speech for interviewer replies

## Tech stack

- Python + Flask
- LangGraph compiled StateGraph (conditional-edge routing, SQLite checkpoints)
- SQLite session persistence
- Groq LLM API (model from `GROQ_MODEL`, default `openai/gpt-oss-20b`)
- ChromaDB + sentence-transformers for resume retrieval
- React + Vite frontend (JavaScript)

## Project structure

```text
interview_practice_partner/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── graph/
│   │   ├── graph.py
│   │   ├── llm.py
│   │   ├── nodes.py
│   │   ├── prompts.py
│   │   └── state.py
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── extractor.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── tests/
│   └── storage.py
├── frontend/              # Vite + React app
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── api/client.js
│   │   ├── hooks/
│   │   └── components/
│   ├── package.json
│   └── vite.config.js
├── README.md
├── PROJECT_EXPLANATION.md
└── .env.example
```

## Quick start

### 1) Backend: virtual environment and dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 2) Add your Groq API key

Copy `.env.example` to `.env` and fill in your values:

```env
GROQ_API_KEY=your_key_here
```

Optional overrides (see `.env.example` for the full list): `GROQ_MODEL`,
`CORS_ORIGINS`, `RATE_LIMIT_CHAT_PER_MIN`, `RAG_MAX_DISTANCE`, `DEBUG_RAG`.

### 3) Start the backend

```powershell
cd backend
python app.py
```

The backend runs on `http://localhost:5000`.

### 4) Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173`. It reads the backend URL from
`VITE_API_URL` (see `frontend/.env.example`); the Vite config also provides an
`/api` proxy as a CORS alternative.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + session count |
| POST | `/chat` | Send a message (`session_id`, `message`) — rate-limited |
| POST | `/feedback` | Generate feedback from session or transcript |
| POST | `/new_session` | Create a session; with `{"session_id"}` resets it (state + checkpoints + resume vectors) |
| GET | `/session/<id>` | Transcript + progress, for restoring after refresh |
| POST | `/resume/upload` | Upload PDF/DOCX resume (5 MB max, magic-byte validated) — rate-limited |
| GET | `/resume/status?session_id=...` | `{uploaded, filename, chunks}` |
| DELETE | `/resume/<id>` | Remove a session's resume |

All errors share the shape `{"error": "...", "code": "..."}`.

## Runtime behavior

- The frontend stores the session ID in localStorage and restores the transcript via `GET /session/<id>` after a refresh.
- Each `/chat` request resumes the same compiled graph thread for that session.
- The graph routes responses through conditional edges; the `next_node` field in state is diagnostic only.
- Vague answers get at most 2 follow-up probes per question, then the interview advances.
- A stop request ("end the interview", …) is honored once at least 3 questions are answered.
- Resume vectors are deleted when a resume is removed, a session is reset, or old sessions are evicted.

## Resume-aware interviewing

The app supports PDF and DOCX resume uploads. Once uploaded:

- the resume text is extracted (scanned/empty PDFs are rejected with a clear error)
- it is chunked section-by-section (Experience, Projects, Skills, …) with metadata
- chunks are embedded and stored per session in ChromaDB (re-uploads replace old chunks)
- retrieval builds its query from the role + current question + last answer, filters by session, and drops low-relevance chunks

Questions in the Introduction and Project stages reference real projects/skills from the resume when available, and the LLM is instructed to use only the retrieved context for personal details.

Set `DEBUG_RAG=1` to log retrieved chunks per turn.

## Testing

Run the backend test suite from the repository root:

```powershell
pytest
```

The suite covers role extraction, stop signals, question counting, the minimum-3-questions rule, classifier routing (mocked LLM), the follow-up cap, feedback fallback, PDF/DOCX extraction, chunker metadata, per-session RAG isolation, re-upload replacement, JSON parsing robustness, and endpoint responses.

## Notes

- Only PDF and DOCX resume formats are supported.
- Voice input/output needs Chrome or Edge with microphone permission; text input always works.
- The app is designed for local development and demos; production deployment would still need hosting, environment management, and auth/security hardening.

## Troubleshooting

If the backend cannot start:

- verify `GROQ_API_KEY` is defined
- make sure all Python dependencies were installed
- ensure no other process is already using port 5000

If the frontend cannot reach the backend:

- confirm the Flask server is running on port 5000
- check `VITE_API_URL` in `frontend/.env`, or use the Vite `/api` proxy

## Additional reference

For the deeper architecture and design notes, see [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).
