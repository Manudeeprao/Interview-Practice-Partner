# Interview Practice Partner

An AI mock interview app that runs a role-based interview, adapts question difficulty based on answers, and uses a candidate resume to personalize follow-up questions.

## What it does

- interviews the user in a role-specific workflow
- stores session state in SQLite between requests
- resumes the same compiled LangGraph graph for each session
- classifies answers as GOOD, VAGUE, OFF_TOPIC, or OUT_OF_SCOPE
- raises or lowers difficulty depending on answer quality
- accepts resume uploads in PDF and DOCX format
- retrieves resume context with ChromaDB + embeddings
- generates end-of-interview feedback
- supports browser microphone input and text input

## Tech stack

- Python + Flask
- LangGraph compiled StateGraph
- SQLite session persistence and checkpoints
- Groq LLM API
- ChromaDB + sentence-transformers for resume retrieval
- Vanilla JavaScript frontend

## Project structure

```text
interview_practice_partner/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── chroma_db/
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
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── README.md
├── PROJECT_EXPLANATION.md
└── venv/
```

## Quick start

### 1) Create and activate a virtual environment

```powershell
cd C:\Users\jupal\OneDrive\Desktop\interview_practice_partner
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install backend dependencies

```powershell
pip install -r backend\requirements.txt
```

### 3) Add the Groq API key

Create a `.env` file in the project root or inside `backend/` with:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

> The app reads from environment variables at runtime. The default model is configurable via `GROQ_MODEL`.

### 4) Start the backend

```powershell
cd backend
python app.py
```

The backend runs on:

```text
http://localhost:5000
```

### 5) Start the frontend

In a separate terminal:

```powershell
cd frontend
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Runtime behavior

- A new session is created automatically when the frontend first loads.
- Resume uploads are attached to that session ID.
- Each chat request resumes the same compiled graph thread for that session.
- The graph routes responses through conditional logic instead of a manual Flask dispatcher.
- The app persists the state so the interview can continue across requests.

## Resume-aware interviewing

The app supports PDF and DOCX resume uploads. Once uploaded:

- the resume text is extracted
- it is chunked and embedded
- the chunks are stored per session in ChromaDB
- the retriever is used to ground interview questions and follow-ups in the uploaded resume

This keeps questions more relevant to the candidate's actual background.

## Testing

Run the backend test suite from the repository root:

```powershell
pytest
```

## Notes

- The frontend can use browser speech recognition if the browser allows microphone access.
- Text input remains available as the fallback path.
- Only PDF and DOCX resume formats are supported for upload.
- The app is designed for local development and demos; production deployment would still need hosting, environment management, and auth/security hardening.

## Troubleshooting

If the backend cannot start:

- verify `GROQ_API_KEY` is defined
- make sure all Python dependencies were installed
- ensure no other process is already using port 5000

If the frontend cannot load:

- confirm the static server is running on port 8000
- check that the backend is reachable at `http://localhost:5000/health`

## Additional reference

For the deeper architecture and design notes, see [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).