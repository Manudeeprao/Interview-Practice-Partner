/**
 * ScoreRing — circular readiness score (1–10), color-coded:
 * 7–10 green, 5–6 amber, 1–4 red. Hidden when score is null.
 */
export function ScoreRing({ score }) {
  if (score == null) return null;
  const colorClass = score >= 7 ? 'score-green' : score >= 5 ? 'score-amber' : 'score-red';
  return (
    <div
      className={`score-ring ${colorClass}`}
      role="img"
      aria-label={`Readiness score: ${score} out of 10`}
    >
      <span className="score-value">{score}</span>
      <span className="score-denom">/10</span>
      <span className="score-ring-label">Readiness</span>
    </div>
  );
}
