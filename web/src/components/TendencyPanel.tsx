import { useEffect, useState } from "react";
import { api, type Tendencies } from "../api";

const SEASONS = Array.from({ length: 16 }, (_, index) => 2024 - index);

export function TendencyPanel({ enabled }: { enabled: boolean }) {
  const [season, setSeason] = useState(2024);
  const [data, setData] = useState<Tendencies | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let current = true;
    api
      .tendencies(season)
      .then((result) => current && setData(result))
      .catch((thrown: Error) => current && setError(thrown.message));
    return () => {
      current = false;
    };
  }, [season, enabled]);

  if (!enabled) {
    return <p className="error">No warehouse loaded. Run `fourthdown build` and restart.</p>;
  }

  const passRate = data?.columns.indexOf("pass_rate") ?? -1;
  const epa = data?.columns.indexOf("epa_per_play") ?? -1;

  return (
    <section>
      <label>
        Season
        <select value={season} onChange={(event) => setSeason(Number(event.target.value))}>
          {SEASONS.map((year) => (
            <option key={year} value={year}>
              {year}
            </option>
          ))}
        </select>
      </label>
      {error && <p className="error">{error}</p>}
      {data && data.rows.length === 0 && <p className="muted">No plays for {data.season}.</p>}
      {data && data.rows.length > 0 && (
        <ul className="options tendencies">
          {data.rows.map((row) => (
            <li key={String(row[0])}>
              <span className="option-name">{String(row[0])}</span>
              <span className="bar" style={{ width: `${Number(row[passRate]) * 100}%` }} />
              <span className="value">{(Number(row[passRate]) * 100).toFixed(0)}% pass</span>
              <span className="muted detail">{Number(row[epa]).toFixed(3)} EPA/play</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
