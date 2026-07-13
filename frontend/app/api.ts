const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ModelPrediction = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
};

export type AnalyzeResponse = {
  text: string;
  prediction: ModelPrediction;
  model_used: string;
  response?: string;
  disclaimer: string;
};

export type CompareResponse = {
  text: string;
  predictions: Record<string, ModelPrediction>;
};

export type InterpretResponse = {
  text: string;
  tokens: string[];
  attention_weights: number[];
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

export function analyze(text: string) {
  return post<AnalyzeResponse>("/analyze", { text, generate_response: true });
}

export function compare(text: string) {
  return post<CompareResponse>("/compare", { text, generate_response: false });
}

export function interpret(text: string) {
  return post<InterpretResponse>("/interpret", { text, generate_response: false });
}

export async function leaderboard() {
  const res = await fetch(`${API_URL}/leaderboard`);
  if (!res.ok) throw new Error("Failed to fetch leaderboard");
  return res.json();
}
