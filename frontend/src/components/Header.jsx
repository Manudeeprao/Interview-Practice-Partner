/**
 * Header — brand, backend health indicator, TTS mute toggle,
 * and the "New interview" button.
 */
export function Header({
  health,
  ttsSupported,
  ttsEnabled,
  onToggleTts,
  onNewInterview,
  sessionId,
}) {
  return (
    <header className="header" aria-label="Application header">
      <div className="header-brand">
        <div className="header-icon" aria-hidden="true">
          🎯
        </div>
        <div>
          <h1 className="header-title">Interview Practice Partner</h1>
          <p className="header-subtitle">AI-powered mock interviews with instant feedback</p>
        </div>
      </div>

      <div className="header-actions">
        <span
          className={`health-dot ${health === 'online' ? 'online' : health === 'offline' ? 'offline' : ''}`}
          role="status"
          aria-label={`Backend ${health}`}
          title={
            health === 'online'
              ? 'Backend is reachable'
              : health === 'offline'
                ? 'Backend is unreachable — start the Flask server'
                : 'Checking backend…'
          }
        >
          <span className="dot" aria-hidden="true" />
          {health === 'online' ? 'API live' : health === 'offline' ? 'API down' : 'API…'}
        </span>

        <button
          type="button"
          className={`header-btn ${ttsEnabled ? 'toggled' : ''}`}
          onClick={onToggleTts}
          disabled={!ttsSupported}
          aria-pressed={ttsEnabled}
          title={
            !ttsSupported
              ? 'Text-to-speech is not supported in this browser'
              : ttsEnabled
                ? 'Mute the interviewer voice'
                : 'Unmute the interviewer voice'
          }
        >
          <span aria-hidden="true">{ttsEnabled ? '🔊' : '🔇'}</span>
          {ttsEnabled ? 'Voice on' : 'Muted'}
        </button>

        <button
          type="button"
          className="header-btn"
          onClick={onNewInterview}
          title="Start a brand-new interview session"
        >
          <span aria-hidden="true">🆕</span> New interview
        </button>
      </div>

      {sessionId && (
        <span className="sr-only">Session {sessionId}</span>
      )}
    </header>
  );
}
