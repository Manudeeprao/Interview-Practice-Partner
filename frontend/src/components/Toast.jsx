/**
 * Toast — fixed-position notification stack (aria-live="assertive").
 */
export function Toast({ toasts, onDismiss }) {
  if (!toasts || toasts.length === 0) return null;
  return (
    <div
      className="toast-container"
      role="status"
      aria-live="assertive"
      aria-atomic="false"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast ${t.type}`}
          onClick={() => onDismiss(t.id)}
          role="button"
          tabIndex={0}
          aria-label={`Dismiss notification: ${t.message}`}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') onDismiss(t.id);
          }}
        >
          {t.message}
        </div>
      ))}
    </div>
  );
}
