import { useEffect, useRef } from 'react';
import { MicButton } from './MicButton';

/**
 * MessageInput — textarea + send button + mic button.
 * Enter sends, Shift+Enter inserts a newline. The interim speech
 * transcript is shown as an overlay inside the input while recording.
 */
export function MessageInput({
  value,
  onChange,
  onSend,
  disabled,
  interim,
  listening,
  mic,
  onMicToggle,
}) {
  const textareaRef = useRef(null);

  // Auto-grow the textarea up to 120px.
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
    }
  }, [value]);

  const canSend = value.trim() !== '' && !disabled;

  const handleSend = () => {
    if (!canSend) return;
    onSend();
    // Return focus to the input so the user can keep typing.
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="input-row">
      <div className="text-input-wrapper">
        <textarea
          ref={textareaRef}
          className="message-textarea"
          rows={1}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            listening
              ? 'Listening… speak now, or type instead'
              : 'Type your answer here, or use the mic button… (Shift+Enter for new line)'
          }
          aria-label="Text input for your answer"
          aria-describedby="input-hint"
        />
        {listening && interim && value.trim() === '' && (
          <div
            className="message-textarea"
            aria-hidden="true"
            style={{
              position: 'absolute',
              inset: 0,
              pointerEvents: 'none',
              color: 'var(--text-muted)',
              fontStyle: 'italic',
              borderColor: 'transparent',
              background: 'transparent',
            }}
          >
            “{interim}”
          </div>
        )}
        <button
          type="button"
          className="send-btn"
          onClick={handleSend}
          disabled={!canSend}
          aria-label="Send message"
          title="Send"
        >
          ↑
        </button>
      </div>

      <MicButton
        supported={mic.supported}
        listening={listening}
        permissionError={mic.permissionError}
        micError={mic.micError}
        disabled={disabled}
        onToggle={onMicToggle}
      />

      <p id="input-hint" className="sr-only">
        Press Enter to send, Shift+Enter for a new line. Use the microphone
        button to speak your answer.
      </p>
    </div>
  );
}
