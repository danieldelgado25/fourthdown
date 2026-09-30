import { useEffect, useState } from "react";
import { api, type Health } from "./api";
import { AdvisorPanel } from "./components/AdvisorPanel";
import { AskPanel } from "./components/AskPanel";
import { TendencyPanel } from "./components/TendencyPanel";

type Tab = "ask" | "advisor" | "tendencies";

const TABS: { id: Tab; label: string }[] = [
  { id: "ask", label: "Assistant" },
  { id: "advisor", label: "Fourth down" },
  { id: "tendencies", label: "Tendencies" },
];

export function App() {
  const [tab, setTab] = useState<Tab>("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((thrown: Error) => setError(thrown.message));
  }, []);

  return (
    <main>
      <header className="top">
        <h1>FourthDown</h1>
        <p className="muted">
          NFL play-by-play 2009-2024 · guarded SQL, hybrid retrieval, and win-probability models
        </p>
        {error && <p className="error">API unreachable: {error}</p>}
        {health && <HealthBar health={health} />}
      </header>

      <nav>
        {TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            className={tab === entry.id ? "tab active" : "tab"}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
          </button>
        ))}
      </nav>

      {tab === "ask" && <AskPanel tools={health?.tools ?? []} />}
      {tab === "advisor" && <AdvisorPanel enabled={health?.models ?? false} />}
      {tab === "tendencies" && <TendencyPanel enabled={health?.warehouse ?? false} />}
    </main>
  );
}

function HealthBar({ health }: { health: Health }) {
  const parts: [string, boolean][] = [
    ["warehouse", health.warehouse],
    ["llm", health.llm],
    ["retrieval", health.retrieval],
    ["models", health.models],
  ];
  return (
    <ul className="health" aria-label="service readiness">
      {parts.map(([name, up]) => (
        <li key={name} className={up ? "up" : "down"} title={health.skipped[name] ?? ""}>
          {name}
        </li>
      ))}
    </ul>
  );
}
