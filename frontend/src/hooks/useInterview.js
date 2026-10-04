import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ApiError,
  createSession,
  getResumeStatus,
  getSession,
  sendChat,
} from '../api/client';

const SESSION_STORAGE_KEY = 'ipp-session-id';
const STOP_PHRASE = "I'm done with the interview, please generate my feedback.";

const GREETING_MESSAGE =
  "Hello! I'm your AI Interview Practice Partner. 🎯\n\n" +
  "Tell me what job role you'd like to practice for — for example: " +
  "'Software Engineer', 'Product Manager', 'Data Analyst', or any role you have in mind.\n\n" +
  "You can speak using the mic button 🎤 or type below.";

let nextMessageId = 1;
const makeMessage = (role, content) => ({ id: nextMessageId++, role, content });

/** Map backend history entries ({role: 'user'|'assistant'}) to UI messages. */
function historyToMessages(history = []) {
  return history.map((entry) =>
    makeMessage(entry.role === 'user' ? 'user' : 'agent', entry.content ?? ''),
  );
}

/**
 * useInterview — owns the whole interview conversation lifecycle.
 *
 * - Persists session_id in localStorage and restores the transcript from
 *   GET /session/<id> on load (falls back to POST /new_session).
 * - sendMessage: optimistic user bubble, typing indicator, retry-with-backoff
 *   via the API client, TTS of interviewer replies.
 * - endInterview / newInterview helpers.
 * - Resume state ({uploaded, filename, chunks}) is owned here so the
 *   ResumeUpload component stays presentational.
 */
export function useInterview({ notify, speak, cancelSpeech }) {
  const [sessionId, setSessionId] = useState(null);
  const [restoring, setRestoring] = useState(true);
  const [messages, setMessages] = useState([]);
  const [thinking, setThinking] = useState(false);
  const [stateInfo, setStateInfo] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [done, setDone] = useState(false);
  const [resume, setResume] = useState({ uploaded: false, filename: null, chunks: 0 });

  const sessionIdRef = useRef(null);
  const notifyRef = useRef(notify);
  const speakRef = useRef(speak);
  const cancelSpeechRef = useRef(cancelSpeech);
  notifyRef.current = notify;
  speakRef.current = speak;
  cancelSpeechRef.current = cancelSpeech;

  const rememberSession = useCallback((id) => {
    sessionIdRef.current = id;
    setSessionId(id);
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, id);
    } catch {
      /* storage unavailable */
    }
  }, []);

  const applyChatResult = useCallback((data) => {
    const agentText = data.response || "I'm here. Please go ahead.";
    setMessages((prev) => [...prev, makeMessage('agent', agentText)]);
    if (data.state_info) {
      setStateInfo(data.state_info);
      if (data.state_info.resume_uploaded) {
        setResume((prev) => ({ ...prev, uploaded: true }));
      }
    }
    if (data.done && data.feedback) {
      setDone(true);
      setFeedback(data.feedback);
      cancelSpeechRef.current?.();
    } else {
      speakRef.current?.(agentText);
    }
  }, []);

  const handleChatError = useCallback((err) => {
    const friendly =
      err instanceof ApiError
        ? err.message
        : 'Something went wrong. Please try again.';
    notifyRef.current?.(friendly, 'error', 8000);
    setMessages((prev) => [
      ...prev,
      makeMessage('agent', "I'm sorry, something went wrong. Please try again."),
    ]);
  }, []);

  /** Initialise: restore a stored session or start a fresh one. */
  useEffect(() => {
    let cancelled = false;

    async function init() {
      let stored = null;
      try {
        stored = localStorage.getItem(SESSION_STORAGE_KEY);
      } catch {
        /* ignore */
      }

      if (stored) {
        try {
          const data = await getSession(stored);
          if (cancelled) return;
          rememberSession(data.session_id || stored);
          setMessages(historyToMessages(data.history));
          setStateInfo(data.state_info || null);
          setDone(Boolean(data.done));
          setFeedback(data.feedback || null);
          try {
            const status = await getResumeStatus(data.session_id || stored);
            if (cancelled) return;
            if (status.uploaded) {
              setResume({
                uploaded: true,
                filename: status.filename || null,
                chunks: status.chunks || 0,
              });
            }
          } catch {
            /* resume status is best-effort on restore */
          }
          setRestoring(false);
          return;
        } catch (err) {
          // Unknown/expired session id — fall through and create a new one.
          if (!(err instanceof ApiError && err.status === 404)) {
            notifyRef.current?.(
              'Could not restore your previous session. Starting a new one.',
              'warning',
            );
          }
        }
      }

      try {
        const data = await createSession();
        if (cancelled) return;
        rememberSession(data.session_id);
        setMessages([makeMessage('agent', GREETING_MESSAGE)]);
      } catch (err) {
        if (cancelled) return;
        handleChatError(err);
        setMessages([makeMessage('agent', GREETING_MESSAGE)]);
      } finally {
        if (!cancelled) setRestoring(false);
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [rememberSession, handleChatError]);

  const sendMessage = useCallback(
    async (text) => {
      const trimmed = (text || '').trim();
      const id = sessionIdRef.current;
      if (!trimmed || thinking || done || restoring || !id) return;

      cancelSpeechRef.current?.();
      setMessages((prev) => [...prev, makeMessage('user', trimmed)]);
      setThinking(true);

      try {
        const data = await sendChat(id, trimmed);
        applyChatResult(data);
      } catch (err) {
        handleChatError(err);
      } finally {
        setThinking(false);
      }
    },
    [thinking, done, restoring, applyChatResult, handleChatError],
  );

  const endInterview = useCallback(async () => {
    await sendMessage(STOP_PHRASE);
  }, [sendMessage]);

  /** Start over: new server session, fresh local state. */
  const newInterview = useCallback(async () => {
    cancelSpeechRef.current?.();
    setThinking(false);
    setDone(false);
    setFeedback(null);
    setStateInfo(null);
    setResume({ uploaded: false, filename: null, chunks: 0 });
    try {
      const data = await createSession();
      rememberSession(data.session_id);
      setMessages([makeMessage('agent', GREETING_MESSAGE)]);
      notifyRef.current?.('New session started! Tell me what role you want to practice.', 'success');
    } catch (err) {
      handleChatError(err);
    }
  }, [rememberSession, handleChatError]);

  const interviewStarted = Boolean(stateInfo?.role_confirmed);

  return {
    sessionId,
    restoring,
    messages,
    thinking,
    stateInfo,
    feedback,
    done,
    interviewStarted,
    resume,
    setResume,
    sendMessage,
    endInterview,
    newInterview,
  };
}
