import { useState } from "react";
import { api, type Advice, type Situation } from "../api";

const INITIAL: Situation = {
  yardline_100: 38,
  ydstogo: 2,
  game_seconds_remaining: 240,
  score_differential: -3,
  posteam_is_home: true,
  posteam_spread: 0,
};

const LABELS: Record<string, string> = {
  go: "Go for it",
  field_goal: "Field goal",
  punt: "Punt",
};

export function AdvisorPanel({ enabled }: { enabled: boolean }) {
  const [situation, setSituation] = useState(INITIAL);
  const [advice, setAdvice] = useState<Advice | null>(null);
  const [error, setError] = useState<string | null>(null);

  function update(field: keyof Situation, value: number) {
    setSituation((current) => ({ ...current, [field]: value }));
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      setAdvice(await api.advise(situation));
    } catch (thrown) {
      setError(thrown instanceof Error ? thrown.message : String(thrown));
      setAdvice(null);
    }
  }

  if (!enabled) {
    return <p className="error">No trained models loaded. Run `fourthdown train` and restart.</p>;
  }

  return (
    <section>
      <form onSubmit={submit} className="grid">
        <Field label="Yards from goal" value={situation.yardline_100} min={1} max={99}
          onChange={(value) => update("yardline_100", value)} />
        <Field label="Yards to go" value={situation.ydstogo} min={0} max={30}
          onChange={(value) => update("ydstogo", value)} />
        <Field label="Seconds left" value={situation.game_seconds_remaining} min={0} max={3600}
          onChange={(value) => update("game_seconds_remaining", value)} />
        <Field label="Score margin" value={situation.score_differential} min={-40} max={40}
          onChange={(value) => update("score_differential", value)} />
        <Field label="Spread" value={situation.posteam_spread} min={-20} max={20}
          onChange={(value) => update("posteam_spread", value)} />
        <button type="submit">Rank the options</button>
      </form>

      {error && <p className="error">{error}</p>}
      {advice && (
        <article className="answer">
          <header>
            <span className="badge">{LABELS[advice.best] ?? advice.best}</span>
            <span className="muted">
              {advice.situation} · +{(advice.edge * 100).toFixed(1)} win probability points
            </span>
          </header>
          <ul className="options">
            {advice.options.map((option) => (
              <li key={option.name}>
                <span className="option-name">{LABELS[option.name] ?? option.name}</span>
                <span className="bar" style={{ width: `${option.win_probability * 100}%` }} />
                <span className="value">{(option.win_probability * 100).toFixed(1)}%</span>
                <span className="muted detail">
                  {option.success_probability !== null &&
                    `${(option.success_probability * 100).toFixed(0)}% success · `}
                  {option.detail}
                </span>
              </li>
            ))}
          </ul>
        </article>
      )}
    </section>
  );
}

function Field({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
