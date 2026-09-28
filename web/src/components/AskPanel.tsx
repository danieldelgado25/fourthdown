import { useState } from "react";
import { api, type AnswerData } from "../api";
import { ResultTable } from "./ResultTable";

const EXAMPLES = [
  "Which team had the highest EPA per play in 2023?",
  "What happened in the 2017 Super Bowl between the Patriots and the Falcons?",
  "Should they go for it on 4th and 2 from the 38, down 3 with four minutes left?",
  "How pass heavy were the Lions in 2023?",
];

export function AskPanel({ tools }: { tools: string[] }) {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [tool, setTool] = useState("");
  const [answer, setAnswer] = useState<AnswerData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setPending(true);
    setError(null);
    try {
      setAnswer(await api.ask(question, tool || undefined));
    } catch (thrown) {
      setError(thrown instanceof Error ? thrown.message : String(thrown));
      setAnswer(null);
    } finally {
      setPending(false);
    }
  }

  return (
    <section>
      <form onSubmit={submit}>
        <label htmlFor="question">Ask anything about 2009-2024 play-by-play</label>
        <textarea
          id="question"
          value={question}
          rows={3}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <div className="row">
          <select
            aria-label="tool"
            value={tool}
            onChange={(event) => setTool(event.target.value)}
          >
            <option value="">route automatically</option>
            {tools.map((name) => (
              <option key={name} value={name}>
                force {name}
              </option>
            ))}
          </select>
          <button type="submit" disabled={pending || question.trim().length === 0}>
            {pending ? "thinking…" : "Ask"}
          </button>
        </div>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button key={example} type="button" className="chip" onClick={() => setQuestion(example)}>
            {example}
          </button>
        ))}
      </div>

      {error && <p className="error">{error}</p>}
      {answer && <AnswerView answer={answer} />}
    </section>
  );
}

function AnswerView({ answer }: { answer: AnswerData }) {
  return (
    <article className={answer.failed ? "answer failed" : "answer"}>
      <header>
        <span className="badge">{answer.tool}</span>
        <span className="muted">
          {answer.route.decided_by}: {answer.route.reason} · {answer.elapsed_seconds.toFixed(2)}s
        </span>
      </header>
      <p className="prose">{answer.answer}</p>
      {answer.table && <ResultTable table={answer.table} />}
      {answer.passages.length > 0 && (
        <ol className="passages">
          {answer.passages.map((passage, index) => (
            <li key={index}>
              <strong>{passage.title}</strong> <span className="muted">({passage.found_by})</span>
              <p>{passage.text}</p>
            </li>
          ))}
        </ol>
      )}
      {answer.detail.sql && (
        <details open>
          <summary>SQL that produced this</summary>
          <pre>{answer.detail.sql}</pre>
        </details>
      )}
    </article>
  );
}
