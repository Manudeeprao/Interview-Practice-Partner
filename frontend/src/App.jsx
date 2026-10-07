import { useCallback, useEffect, useState } from 'react';
import { getHealth } from './api/client';
import { useInterview } from './hooks/useInterview';
import { useSpeechRecognition } from './hooks/useSpeechRecognition';
import { useSpeechSynthesis } from './hooks/useSpeechSynthesis';
import { useToasts } from './hooks/useToasts';
import { Header } from './components/Header';
import { ChatWindow } from './components/ChatWindow';
import { MessageInput } from './components/MessageInput';
import { ResumeUpload } from './components/ResumeUpload';
import { ProgressBar } from './components/ProgressBar';
import { FeedbackPanel } from './components/FeedbackPanel';
import { EndInterviewButton } from './components/EndInterviewButton';
import { Toast } from './components/Toast';

const HEALTH_POLL_MS = 15000;

function App() {
  const { toasts, push: notify, dismiss: dismissToast } = useToasts();
  const tts = useSpeechSynthesis();
  const interview = useInterview({
    notify,
    speak: tts.speak,
    cancelSpeech: tts.cancel,
  });

  const [inputText, setInputText] = useState('');
  const [health, setHealth] = useState('checking');

  const {
    sessionId,
    restoring,
    messages,
    thinking,
    stateInfo,
    feedback,
    done,
    resume,
    setResume,
    sendMessage,
    endInterview,
    newInterview,
  } = interview;

  // ── Backend health polling ───────────────────────────────────
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        await getHealth();
        if (alive) setHealth('online');
      } catch {
        if (alive) setHealth('offline');
      }
    };
    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  // ── Speech recognition → dictate into the answer box ──────────
  // Final transcripts accumulate into the input so the candidate can review
  // and edit before sending — fragments are never auto-sent.
  const recognition = useSpeechRecognition({
    onFinal: (fullText) => {
      setInputText(fullText);
    },
  });

  // Surface mic problems as toasts instead of failing silently.
  useEffect(() => {
    if (recognition.micError) {
      notify(recognition.micError, 'warning', 6000);
    }
  }, [recognition.micError, notify]);

  const handleMicToggle = useCallback(() => {
    if (recognition.listening) {
      recognition.stop();
    } else {
      // Never let the interviewer talk over the user.
      tts.cancel();
      recognition.start();
    }
  }, [recognition, tts]);

  // Ctrl/Cmd + M toggles the mic.
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'm') {
        e.preventDefault();
        if (!thinking && !done) handleMicToggle();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [handleMicToggle, thinking, done]);

  // ── Actions ──────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    sendMessage(inputText);
    setInputText('');
  }, [sendMessage, inputText]);

  const handleSuggestRole = useCallback(
    (role) => {
      sendMessage(role);
    },
    [sendMessage],
  );

  const handleNewInterview = useCallback(() => {
    // Guard against accidentally wiping an interview in progress.
    if (messages.length > 1 && !done) {
      const confirmed = window.confirm(
        'Start a new interview? Your current conversation will be discarded.',
      );
      if (!confirmed) return;
    }
    setInputText('');
    newInterview();
  }, [messages.length, done, newInterview]);

  const handleEndInterview = useCallback(() => {
    endInterview();
  }, [endInterview]);

  // ── Status badge ─────────────────────────────────────────────
  const status = done
    ? 'done'
    : thinking
      ? 'thinking'
      : recognition.listening
        ? 'listening'
        : tts.speaking
          ? 'speaking'
          : 'idle';

  const inputDisabled = thinking || done || restoring || !sessionId;
  const questionCount = stateInfo?.main_question_count ?? 0;

  return (
    <div className="app-container">
      <Header
        health={health}
        ttsSupported={tts.supported}
        ttsEnabled={tts.enabled}
        onToggleTts={() => tts.setEnabled(!tts.enabled)}
        onNewInterview={handleNewInterview}
        sessionId={sessionId}
      />

      {health === 'offline' && (
        <div
          className="toast error"
          role="alert"
          style={{ maxWidth: 'none' }}
        >
          Cannot reach the backend. Make sure the Flask server is running
          (python app.py in backend/).
        </div>
      )}

      <ChatWindow
        messages={messages}
        thinking={thinking}
        status={status}
        showSuggestions={!done && !restoring && !thinking}
        onSuggestRole={handleSuggestRole}
      />

      {!done && (
        <div className="input-zone" aria-label="Input controls">
          <MessageInput
            value={inputText}
            onChange={setInputText}
            onSend={handleSend}
            disabled={inputDisabled}
            interim={recognition.interim}
            listening={recognition.listening}
            mic={recognition}
            onMicToggle={handleMicToggle}
          />

          <div className="controls-row">
            <ProgressBar stateInfo={stateInfo} />
            <ResumeUpload
              sessionId={sessionId}
              resume={resume}
              setResume={setResume}
              notify={notify}
            />
            <EndInterviewButton
              questionCount={questionCount}
              disabled={inputDisabled && !thinking}
              onEnd={handleEndInterview}
            />
          </div>

          {recognition.listening && recognition.interim === '' && (
            <span className="interim-text" aria-live="polite">
              Listening…
            </span>
          )}
        </div>
      )}

      {done && feedback && (
        <FeedbackPanel
          feedback={feedback}
          onRestart={handleNewInterview}
          notify={notify}
        />
      )}

      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

export default App;
