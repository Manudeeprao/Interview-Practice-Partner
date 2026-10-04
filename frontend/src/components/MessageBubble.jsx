/**
 * MessageBubble — a single chat bubble. Renders **bold** markdown
 * minimally (built as an element tree, so no HTML injection risk).
 */
function renderRichText(text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    const match = part.match(/^\*\*([^*]+)\*\*$/);
    if (match) {
      return <strong key={i}>{match[1]}</strong>;
    }
    return <span key={i}>{part}</span>;
  });
}

export function MessageBubble({ message }) {
  const isAgent = message.role === 'agent';
  return (
    <div className={`message ${isAgent ? 'agent' : 'user'}`}>
      <div className="avatar" aria-hidden="true">
        {isAgent ? '🤖' : '👤'}
      </div>
      <div className="bubble">
        <div className="bubble-label">{isAgent ? 'Interviewer' : 'You'}</div>
        <div className="bubble-text">{renderRichText(message.content)}</div>
      </div>
    </div>
  );
}

/** Animated three-dot "the interviewer is typing" bubble. */
export function TypingBubble() {
  return (
    <div className="message agent" aria-hidden="true">
      <div className="avatar">🤖</div>
      <div className="bubble">
        <div className="bubble-label">Interviewer</div>
        <div className="typing-indicator">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
