/**
 * Interview Practice Partner — backend API client.
 *
 * Backend URL comes from import.meta.env.VITE_API_URL and defaults to
 * http://localhost:5000. Set VITE_API_URL=/api to route through the
 * Vite dev-server proxy instead (a CORS alternative).
 */

const RAW_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000';
export const API_BASE = RAW_BASE.replace(/\/+$/, '');

const MAX_ATTEMPTS = 3;
const BASE_BACKOFF_MS = 1000;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Error thrown for HTTP / network failures. Carries status + backend code. */
export class ApiError extends Error {
  constructor(message, { status = 0, code = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/**
 * fetch with exponential backoff. Retries network failures, HTTP 429
 * (rate-limited) and 5xx responses, up to MAX_ATTEMPTS total attempts.
 */
async function fetchWithRetry(url, options = {}, attempts = MAX_ATTEMPTS) {
  let lastError = null;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, options);

      if (response.status === 429 || response.status >= 500) {
        // Retryable HTTP status — read body for a friendly message, then back off.
        let body = {};
        try {
          body = await response.json();
        } catch {
          /* non-JSON body: fall through to the status-based message */
        }
        const message =
          body.error ||
          (response.status === 429
            ? 'Rate limited by the server. Retrying…'
            : `Server error (${response.status}). Retrying…`);
        lastError = new ApiError(message, { status: response.status, code: body.code || null });
        if (attempt < attempts - 1) {
          await delay(BASE_BACKOFF_MS * 2 ** attempt);
          continue;
        }
        throw lastError;
      }

      if (!response.ok) {
        let body = {};
        try {
          body = await response.json();
        } catch {
          /* ignore */
        }
        throw new ApiError(
          body.error || `Request failed (${response.status})`,
          { status: response.status, code: body.code || null },
        );
      }

      return response;
    } catch (err) {
      if (err instanceof ApiError) {
        // Non-retryable HTTP error (4xx other than 429) — surface immediately.
        throw err;
      }
      // Network failure (DNS, connection refused, offline…): retry.
      lastError = new ApiError(
        'Cannot reach the backend — is the Flask server running?',
        { status: 0 },
      );
      if (attempt < attempts - 1) {
        await delay(BASE_BACKOFF_MS * 2 ** attempt);
      }
    }
  }

  throw lastError;
}

async function postJson(path, payload) {
  const response = await fetchWithRetry(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}

async function getJson(path) {
  const response = await fetchWithRetry(`${API_BASE}${path}`);
  return response.json();
}

/** GET /health → {status, sessions} */
export function getHealth() {
  return getJson('/health');
}

/**
 * POST /new_session. Pass a session_id to reset that session server-side,
 * or call with no arguments for a brand-new session.
 */
export function createSession(sessionId) {
  return postJson('/new_session', sessionId ? { session_id: sessionId } : {});
}

/** POST /chat → {session_id, response, done, feedback, state_info} */
export function sendChat(sessionId, message) {
  return postJson('/chat', { session_id: sessionId, message });
}

/** POST /feedback → {feedback} */
export function requestFeedback(sessionId, transcript = [], role = null) {
  const payload = { session_id: sessionId, transcript };
  if (role) payload.role = role;
  return postJson('/feedback', payload);
}

/** GET /session/<id> → transcript + state restore payload (404 if unknown). */
export function getSession(sessionId) {
  return getJson(`/session/${encodeURIComponent(sessionId)}`);
}

/** GET /resume/status → {uploaded, filename, chunks, session_id} */
export function getResumeStatus(sessionId) {
  return getJson(`/resume/status?session_id=${encodeURIComponent(sessionId)}`);
}

/** DELETE /resume/<session_id> → {session_id, deleted, message} */
export async function deleteResume(sessionId) {
  const response = await fetchWithRetry(
    `${API_BASE}/resume/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE' },
  );
  return response.json();
}

/**
 * POST /resume/upload (multipart). Uses XMLHttpRequest so we can report
 * upload progress. Resolves with the parsed JSON body.
 */
export function uploadResume(sessionId, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/resume/upload`);

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && typeof onProgress === 'function') {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });

    xhr.addEventListener('load', () => {
      let body = {};
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      } catch {
        body = { error: 'The server returned an unreadable response.' };
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
      } else {
        reject(
          new ApiError(body.error || `Upload failed (${xhr.status})`, {
            status: xhr.status,
            code: body.code || null,
          }),
        );
      }
    });

    xhr.addEventListener('error', () => {
      reject(new ApiError('Cannot reach the backend — is the Flask server running?', { status: 0 }));
    });
    xhr.addEventListener('abort', () => {
      reject(new ApiError('Upload was cancelled.', { status: 0 }));
    });

    const form = new FormData();
    form.append('file', file);
    form.append('session_id', sessionId);
    xhr.send(form);
  });
}
