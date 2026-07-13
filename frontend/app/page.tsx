"use client";

import { useState } from "react";
import { analyze, AnalyzeResponse } from "./api";

export default function AnalyzePage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyze(text);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Analyze</h1>
      <p className="subtitle">
        Paste a piece of text. MindGuard classifies it and generates a supportive reply.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type or paste text here..."
      />
      <button onClick={handleSubmit} disabled={loading || !text.trim()}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card">
          <span className="pill">{result.prediction.label}</span>
          <span style={{ marginLeft: 10, color: "var(--muted)", fontSize: 13 }}>
            {(result.prediction.confidence * 100).toFixed(1)}% confidence · model: {result.model_used}
          </span>

          <div style={{ marginTop: 16 }}>
            {Object.entries(result.prediction.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([label, prob]) => (
                <div className="bar-row" key={label}>
                  <span className="bar-label">{label}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span>{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>

          {result.response && (
            <div style={{ marginTop: 20 }}>
              <div style={{ color: "var(--muted)", fontSize: 13, marginBottom: 6 }}>
                Empathetic response
              </div>
              <div style={{ lineHeight: 1.6 }}>{result.response}</div>
            </div>
          )}

          <div className="disclaimer">{result.disclaimer}</div>
        </div>
      )}
    </div>
  );
}
