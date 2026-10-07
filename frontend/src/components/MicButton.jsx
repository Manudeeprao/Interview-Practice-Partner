/**
 * MicButton — toggles speech recognition.
 * Disabled with a clear note when the Web Speech API is unavailable
 * (Chrome is required for voice input).
 */
export function MicButton({
  supported,
  listening,
  permissionError,
  micError,
  disabled,
  onToggle,
}) {
  if (!supported || permissionError) {
    return (
      <div className="mic-area">
        <button
          type="button"
          className="mic-btn"
          disabled
          aria-label="Voice input unavailable"
          title="Voice input requires Chrome. Please type your answers instead."
        >
          🎤
        </button>
        <span className="mic-label" aria-hidden="true">
          Mic
        </span>
        <span className="mic-unsupported-note">
          Voice needs Chrome — please type instead.
        </span>
      </div>
    );
  }

  return (
    <div className="mic-area">
      <button
        type="button"
        className={`mic-btn${listening ? ' recording' : ''}`}
        onClick={onToggle}
        disabled={disabled}
        aria-pressed={listening}
        aria-label={listening ? 'Stop recording' : 'Start voice recording'}
        title={listening ? 'Click to stop — your words stay in the box for review' : 'Click to dictate your answer (Ctrl+M)'}
      >
        🎤
      </button>
      <span className="mic-label" aria-hidden="true">
        {listening ? 'Stop' : 'Mic'}
      </span>
      {micError && !listening && (
        <span className="mic-error-note" role="alert">
          {micError}
        </span>
      )}
    </div>
  );
}
