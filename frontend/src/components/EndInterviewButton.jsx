const MIN_QUESTIONS_TO_END = 3;

/**
 * EndInterviewButton — disabled until at least 3 questions have been
 * answered (the backend also enforces this). Asks for confirmation,
 * then the parent sends the stop phrase through /chat.
 */
export function EndInterviewButton({ questionCount, disabled, onEnd }) {
  const canEnd = (questionCount ?? 0) >= MIN_QUESTIONS_TO_END;

  const handleClick = () => {
    if (!canEnd) return;
    const confirmed = window.confirm(
      'End the interview now and generate your feedback report?',
    );
    if (confirmed) onEnd();
  };

  return (
    <button
      type="button"
      className="btn-end-interview"
      onClick={handleClick}
      disabled={disabled || !canEnd}
      aria-label="End interview and generate feedback"
      title={
        canEnd
          ? 'End the interview and generate your feedback'
          : `Answer at least ${MIN_QUESTIONS_TO_END} questions before ending the interview`
      }
    >
      <span aria-hidden="true">⏹</span> End Interview
    </button>
  );
}
