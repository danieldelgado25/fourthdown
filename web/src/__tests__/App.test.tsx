import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import type { Advice, AnswerData, Health, Tendencies } from "../api";

const HEALTH: Health = {
  status: "ok",
  warehouse: true,
  llm: true,
  retrieval: false,
  models: true,
  tools: ["stats", "narrative", "advisor"],
  skipped: { retrieval: "postgres unreachable" },
};

const SQL_ANSWER: AnswerData = {
  question: "epa leaders in 2023",
  tool: "stats",
  route: { reason: "statistical question", decided_by: "keyword" },
  answer: "SF led at 0.161 EPA per play.",
  table: { columns: ["team", "epa"], rows: [["SF", 0.161]] },
  passages: [],
  detail: { sql: "SELECT posteam FROM plays" },
  failed: false,
  elapsed_seconds: 1.2,
};

const NARRATIVE_ANSWER: AnswerData = {
  ...SQL_ANSWER,
  tool: "narrative",
  table: null,
  detail: {},
  answer: "The Patriots erased a 28-3 deficit.",
  passages: [
    { title: "NE vs ATL, 2016 postseason", text: "28-3 comeback", score: 0.98, found_by: "both" },
  ],
};

const ADVICE: Advice = {
  situation: "4th and 2 from the 38",
  best: "go",
  edge: 0.043,
  options: [
    { name: "go", win_probability: 0.51, success_probability: 0.58, detail: "convert or turn over" },
    { name: "punt", win_probability: 0.467, success_probability: null, detail: "net 38 yards" },
  ],
};

const TENDENCIES: Tendencies = {
  season: 2024,
  columns: ["team", "plays", "pass_rate", "early_down_pass_rate", "epa_per_play"],
  rows: [["DET", 700, 0.61, 0.6, 0.12]],
};

function mockFetch(routes: Record<string, unknown>, ok = true) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    void init;
    const url = String(input);
    const key = Object.keys(routes).find((path) => url.startsWith(path));
    return {
      ok: key !== undefined && ok,
      status: key === undefined ? 404 : ok ? 200 : 503,
      json: async () => (key === undefined ? { error: "not found" } : routes[key]),
    } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("shows which backing services are up", async () => {
    mockFetch({ "/api/health": HEALTH });
    render(<App />);
    const readiness = await screen.findByLabelText("service readiness");
    expect(readiness).toHaveTextContent("warehouse");
    expect(readiness.querySelectorAll(".up")).toHaveLength(3);
    expect(readiness.querySelectorAll(".down")).toHaveLength(1);
  });

  it("renders an API error instead of an empty page", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 500, json: async () => ({}) }) as Response),
    );
    render(<App />);
    expect(await screen.findByText(/API unreachable/)).toBeInTheDocument();
  });

  it("renders a SQL-backed answer with its table and provenance", async () => {
    mockFetch({ "/api/health": HEALTH, "/api/ask": SQL_ANSWER });
    render(<App />);
    await screen.findByLabelText("service readiness");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("SF led at 0.161 EPA per play.")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "epa" })).toBeInTheDocument();
    expect(screen.getByText("0.161")).toBeInTheDocument();
    expect(screen.getByText("SELECT posteam FROM plays")).toBeInTheDocument();
    expect(screen.getByText("stats")).toBeInTheDocument();
  });

  it("renders retrieved passages for a narrative answer", async () => {
    mockFetch({ "/api/health": HEALTH, "/api/ask": NARRATIVE_ANSWER });
    render(<App />);
    await screen.findByLabelText("service readiness");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("NE vs ATL, 2016 postseason")).toBeInTheDocument();
    expect(screen.getByText("(both)")).toBeInTheDocument();
  });

  it("forces a tool when one is selected", async () => {
    const fetchMock = mockFetch({ "/api/health": HEALTH, "/api/ask": SQL_ANSWER });
    render(<App />);
    await screen.findByLabelText("service readiness");
    await userEvent.selectOptions(screen.getByLabelText("tool"), "narrative");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const body = JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body));
    expect(body.tool).toBe("narrative");
  });

  it("ranks fourth-down options by win probability", async () => {
    mockFetch({ "/api/health": HEALTH, "/api/advise": ADVICE });
    render(<App />);
    await screen.findByLabelText("service readiness");
    await userEvent.click(screen.getByRole("button", { name: "Fourth down" }));
    await userEvent.click(screen.getByRole("button", { name: "Rank the options" }));

    expect(await screen.findAllByText("Go for it")).toHaveLength(2);
    expect(screen.getByText("51.0%")).toBeInTheDocument();
    expect(screen.getByText(/\+4.3 win probability points/)).toBeInTheDocument();
  });

  it("explains that the advisor needs trained models", async () => {
    mockFetch({ "/api/health": { ...HEALTH, models: false } });
    render(<App />);
    await screen.findByLabelText("service readiness");
    await userEvent.click(screen.getByRole("button", { name: "Fourth down" }));
    expect(screen.getByText(/No trained models loaded/)).toBeInTheDocument();
  });

  it("loads team tendencies for the selected season", async () => {
    const fetchMock = mockFetch({ "/api/health": HEALTH, "/api/tendencies": TENDENCIES });
    render(<App />);
    await screen.findByLabelText("service readiness");
    await userEvent.click(screen.getByRole("button", { name: "Tendencies" }));

    expect(await screen.findByText("DET")).toBeInTheDocument();
    expect(screen.getByText("61% pass")).toBeInTheDocument();
    expect(String(fetchMock.mock.calls[1][0])).toContain("season=2024");
  });
});
