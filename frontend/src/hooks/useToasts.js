import { useCallback, useState } from 'react';

let nextId = 1;

/**
 * useToasts — tiny toast notification store.
 * push(message, type) where type ∈ 'info' | 'success' | 'warning' | 'error'.
 * Toasts auto-dismiss after `duration` ms (default 4s).
 */
export function useToasts() {
  const [toasts, setToasts] = useState([]);

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (message, type = 'info', duration = 4000) => {
      const id = nextId++;
      setToasts((prev) => [...prev, { id, message, type }]);
      if (duration > 0) {
        setTimeout(() => dismiss(id), duration);
      }
      return id;
    },
    [dismiss],
  );

  return { toasts, push, dismiss };
}
