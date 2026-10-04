# Interview Practice Partner — Frontend (React)

Vite + React (JavaScript) frontend for the Interview Practice Partner mock-interview app.

## Setup

```bash
cd frontend
npm install
npm run dev      # starts on http://localhost:5173
```

Other scripts:

```bash
npm run build    # production build → frontend/dist/
npm run preview  # preview the production build locally
```

The Flask backend must be running separately (see the repo root README):

```bash
cd backend
python app.py    # http://localhost:5000
```

## Environment

Copy `.env.example` to `.env` and adjust:

```bash
VITE_API_URL=http://localhost:5000
```

## CORS alternative: the Vite dev proxy

If the browser blocks direct calls to the backend (CORS), you can route API
traffic through the Vite dev server instead of calling the backend directly:

1. Set `VITE_API_URL=/api` in `frontend/.env`.
2. Restart `npm run dev`.

`vite.config.js` already proxies `/api/*` → `http://localhost:5000/*`
(stripping the `/api` prefix), so no backend changes are needed.

## Features

- Chat interview with optimistic messages, typing indicator, and retry with
  exponential backoff on network errors / HTTP 429 / 5xx.
- `session_id` persisted in `localStorage` (`ipp-session-id`); refresh restores
  the transcript via `GET /session/<id>`.
- Voice input via the Web Speech API (Chrome required) with interim
  transcript shown in the input; text-to-speech replies with a mute toggle.
- Resume upload (drag-and-drop or picker, PDF/DOCX, 5 MB limit, progress bar)
  with remove support (`DELETE /resume/<session_id>`).
- Progress bar driven by `state_info` from the backend.
- End-interview flow (minimum 3 questions, confirm dialog) and a feedback
  panel with readiness score, copy-to-clipboard, and restart.
