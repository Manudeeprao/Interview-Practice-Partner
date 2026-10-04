import { ScoreRing } from './ScoreRing';

/**
 * FeedbackPanel — end-of-interview report: readiness score ring,
 * feedback cards, improvement areas, Copy + Restart actions.
 */
export function FeedbackPanel({ feedback, onRestart, notify }) {
  if (!feedback) return null;

  const score = feedback.readinessScore;
  const scoreLabel =
    score == null ? '' : score >= 7 ? 'Strong performance!' : score >= 5 ? 'Good progress!' : 'Room to improve.';

  const cards = [
    { icon: '💡', iconClass: 'blue', title: 'Overall Impression', content: feedback.overallImpression },
    { icon: '🗣️', iconClass: 'teal', title: 'Communication', content: feedback.communication },
    { icon: '🧠', iconClass: 'purple', title: 'Technical Knowledge', content: feedback.technicalKnowledge },
    { icon: '⭐', iconClass: 'green', title: 'Strengths', content: feedback.strengths },
  ];

  const handleCopy = async () => {
    const lines = [
      `${feedback.interviewRole ? `${feedback.interviewRole} ` : ''}Interview Feedback`,
      score != null ? `Job Readiness Score: ${score}/10` : '',
      '',
      '== Overall Impression ==',
      feedback.overallImpression || '',
      '',
      '== Communication ==',
      feedback.communication || '',
      '',
      '== Technical Knowledge ==',
      feedback.technicalKnowledge || '',
      '',
      '== Strengths ==',
      feedback.strengths || '',
      '',
      '== Areas for Improvement ==',
      ...(feedback.improvementAreas || []).map((a) => `• ${a}`),
    ];
    try {
      await navigator.clipboard.writeText(lines.join('\n'));
      notify?.('Feedback copied to clipboard!', 'success');
    } catch {
      notify?.('Could not copy — please select and copy the text manually.', 'warning');
    }
  };

  return (
    <section className="feedback-panel" aria-label="Interview feedback" aria-live="polite">
      <div className="feedback-header">
        <div className="feedback-header-top">
          <div>
            <h2>
              🏆 {feedback.interviewRole ? `${feedback.interviewRole} Interview Feedback` : 'Your Interview Feedback'}
            </h2>
            <p>
              Here&apos;s your personalized performance analysis.
              {score != null && ` Job readiness: ${score}/10 — ${scoreLabel}`}
            </p>
          </div>
          <ScoreRing score={score} />
        </div>
      </div>

      <div className="feedback-grid">
        {cards.map(
          (card) =>
            card.content && (
              <div className="feedback-card" key={card.title}>
                <div className="card-icon-row">
                  <div className={`card-icon ${card.iconClass}`} aria-hidden="true">
                    {card.icon}
                  </div>
                  <span className="card-title">{card.title}</span>
                </div>
                <p className="card-text">{card.content}</p>
              </div>
            ),
        )}

        {feedback.improvementAreas?.length > 0 && (
          <div className="feedback-card full-width" aria-label="Areas for improvement">
            <div className="card-icon-row">
              <div className="card-icon amber" aria-hidden="true">
                📈
              </div>
              <span className="card-title">Areas for Improvement</span>
            </div>
            <div className="improvement-list" role="list">
              {feedback.improvementAreas.map((area, i) => (
                <div className="improvement-item" role="listitem" key={i}>
                  <span className="improvement-bullet" aria-hidden="true">
                    ▸
                  </span>
                  <span>{area}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="feedback-actions">
        <button
          type="button"
          className="btn-copy"
          onClick={handleCopy}
          aria-label="Copy feedback to clipboard"
        >
          <span aria-hidden="true">📋</span> Copy feedback
        </button>
        <button
          type="button"
          className="btn-restart"
          onClick={onRestart}
          aria-label="Start a new interview session"
        >
          <span aria-hidden="true">🔄</span> Restart
        </button>
      </div>
    </section>
  );
}
