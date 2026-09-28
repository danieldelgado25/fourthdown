// The wire format, typed once. Everything the dashboard renders comes from here, and
// nothing is computed in the browser that the API could compute from the warehouse.

export interface Health {
  status: "ok" | "degraded";
  warehouse: boolean;
  llm: boolean;
  retrieval: boolean;
  models: boolean;
  tools: string[];
  skipped: Record<string, string>;
}

export interface TableData {
  columns: string[];
  rows: (string | number | boolean | null)[][];
}

export interface PassageData {
  title: string;
  text: string;
  score: number;
  found_by: string;
}

export interface AnswerData {
  question: string;
  tool: string;
  route: { reason: string; decided_by: string };
  answer: string;
  table: TableData | null;
  passages: PassageData[];
  detail: Record<string, string>;
  failed: boolean;
  elapsed_seconds: number;
}

export interface OptionData {
  name: string;
  win_probability: number;
  success_probability: number | null;
  detail: string;
}

export interface Advice {
  situation: string;
  best: string;
  edge: number;
  options: OptionData[];
}

export interface Tendencies {
  season: number;
  columns: string[];
  rows: (string | number | null)[][];
}

export interface Situation {
  yardline_100: number;
  ydstogo: number;
  game_seconds_remaining: number;
  score_differential: number;
  posteam_is_home: boolean;
  posteam_spread: number;
}

async function send<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((body as { error?: string }).error ?? `request failed (${response.status})`);
  }
  return body as T;
}

export const api = {
  health: () => send<Health>("/api/health"),
  ask: (question: string, tool?: string) =>
    send<AnswerData>("/api/ask", {
      method: "POST",
      body: JSON.stringify(tool ? { question, tool } : { question }),
    }),
  advise: (situation: Situation) =>
    send<Advice>("/api/advise", { method: "POST", body: JSON.stringify(situation) }),
  tendencies: (season: number) => send<Tendencies>(`/api/tendencies?season=${season}`),
};
