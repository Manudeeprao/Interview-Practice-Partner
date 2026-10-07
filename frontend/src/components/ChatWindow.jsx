import { useEffect, useRef } from 'react';
import { MessageBubble, TypingBubble } from './MessageBubble';

const STATUS_LABELS = {
  idle: 'Ready',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
  listening: 'Listening…',
  done: 'Interview Complete',
};

/**
 * ChatWindow — the conversation transcript.
 * role="log" + aria-live="polite" announces new messages to screen readers.
 * Auto-scrolls to the newest message.
 */
export function ChatWindow({ messages, thinking, status, showSuggestions, onSuggestRole }) {
  const logRef = useRef(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, thinking]);

  const suggestions = [
    'Software Engineer',
    'Data Analyst',
    'Product Manager',
    'Data Scientist',
  ];

  return (
    <section className="chat-panel" aria-label="Interview conversation">
      <div className="chat-panel-header">
        <span className="chat-panel-title">Conversation</span>
        <div className="status-bar" aria-live="polite" aria-label="Interview status">
          <div className={`status-badge ${status}`} role="status">
            <span className="status-dot" aria-hidden="true" />
            <span>{STATUS_LABELS[status] || 'Ready'}</span>
          </div>
        </div>
      </div>

      <div
        ref={logRef}
        className="chat-log"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-label="Interview transcript"
      >
        {messages.length === 0 && !thinking ? (
          <div className="chat-empty">
            <div className="chat-empty-icon" aria-hidden="true">
              💬
            </div>
            <p>Your interview conversation will appear here.</p>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} />)
        )}
        {thinking && <TypingBubble />}
        {showSuggestions && messages.length === 1 && !thinking && (
          <div className="chat-empty" style={{ flex: '0 0 auto', paddingTop: 4 }}>
            <p>Pick a role to start — or type your own below.</p>
            <div className="role-chips" aria-label="Suggested roles">
              {suggestions.map((role) => (
                <button
                  key={role}
                  type="button"
                  className="role-chip"
                  onClick={() => onSuggestRole?.(role)}
                >
                  {role}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
