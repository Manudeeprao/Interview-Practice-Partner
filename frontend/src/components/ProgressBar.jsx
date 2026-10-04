const STAGE_LABELS = {
  introduction: 'Introduction',
  project_discussion: 'Projects',
  technical_fundamentals: 'Technical',
  system_design: 'System Design',
  behavioral: 'Behavioral',
  ROLE_SELECTION: 'Setup',
  INTERVIEW_ACTIVE: 'Interview',
  feedback: 'Feedback',
  INTERVIEW_COMPLETE: 'Complete',
};

/**
 * ProgressBar — driven entirely by backend state_info:
 * width = main_question_count / max_questions, plus a stage pill
 * and the current difficulty.
 */
export function ProgressBar({ stateInfo }) {
  const qCount = stateInfo?.main_question_count ?? 0;
  const maxQ = stateInfo?.max_questions ?? 7;
  const stage = stateInfo?.interview_stage || '';
  const difficulty = stateInfo?.difficulty;

  const pct = maxQ > 0 ? Math.min(100, Math.round((qCount / maxQ) * 100)) : 0;
  const stageLabel = STAGE_LABELS[stage] || stage || '';
  const visible = qCount > 0 || Boolean(stateInfo?.role_confirmed);

  return (
    <div
      className={`progress-section${visible ? ' visible' : ''}`}
      aria-label="Interview progress"
    >
      <div className="progress-meta">
        <span className="progress-label">
          Q {qCount} / {maxQ}
        </span>
        {stageLabel && (
          <span className="stage-label">
            {stageLabel}
            {difficulty ? ` · ${String(difficulty).toUpperCase()}` : ''}
          </span>
        )}
      </div>
      <div
        className="progress-bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-label={`Question ${qCount} of ${maxQ}`}
      >
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
