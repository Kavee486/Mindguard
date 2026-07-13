"use client";

import { useState } from "react";
import { interpret, InterpretResponse } from "../api";

function weightToColor(w: number, max: number) {
  const intensity = max > 0 ? Math.min(w / max, 1) : 0;
  const alpha = 0.15 + intensity * 0.7;
  return `rgba(108, 139, 255, ${alpha})`;
}

export default function InterpretPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<InterpretResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await interpret(text);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const maxWeight = result ? Math.max(...result.attention_weights) : 0;

  return (
    <div>
      <h1>Interpretability</h1>
      <p className="subtitle">
        Shows which tokens BERT&apos;s [CLS] representation attends to most (last layer, averaged over heads).
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type or paste text here..."
      />
      <button onClick={handleSubmit} disabled={loading || !text.trim()}>
        {loading ? "Analyzing..." : "Show attention"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card">
          <div style={{ lineHeight: 2.2, fontSize: 15 }}>
            {result.tokens.map((tok, i) => (
              <span
                key={i}
                style={{
                  background: weightToColor(result.attention_weights[i], maxWeight),
                  padding: "2px 4px",
                  borderRadius: 4,
                  marginRight: 2,
                  display: "inline-block",
                }}
              >
                {tok.replace("##", "")}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
