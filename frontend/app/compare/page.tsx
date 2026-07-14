"use client";

import { useState } from "react";
import { compare, CompareResponse } from "../api";
import Avatar from "../Avatar";

export default function ComparePage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await compare(text);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="hero">
        <div className="avatar-glow">
          <Avatar size={52} />
        </div>
        <div>
          <h1>Model Comparison</h1>
          <p className="subtitle">See predictions from every trained model side by side.</p>
        </div>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type or paste text here..."
      />
      <button onClick={handleSubmit} disabled={loading || !text.trim()}>
        {loading ? "Comparing..." : "Compare"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Predicted label</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.predictions).map(([model, pred]) => (
                <tr key={model}>
                  <td>
                    <span className="pill">{model}</span>
                  </td>
                  <td>{pred.label}</td>
                  <td>{(pred.confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
