"use client";

import { useEffect, useRef, useState } from "react";
import { analyze, AnalyzeResponse } from "./api";
import Avatar from "./Avatar";

type ChatMessage =
  | { role: "user"; text: string }
  | { role: "assistant"; result: AnalyzeResponse }
  | { role: "assistant-error"; text: string };

const SUGGESTIONS = [
  "I've been feeling really overwhelmed lately",
  "I can't stop worrying about everything",
  "Today was actually a pretty good day",
];

const GREETING_LINES = [
  "Hi, I'm here to listen 👋",
  "Do you feel okay today?",
  "Need someone to talk to?",
  "Whatever it is, we're here for you.",
];

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path d="M4 12l16-8-6 8 6 8-16-8z" fill="white" />
    </svg>
  );
}

export default function AnalyzePage() {
  const [text, setText] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [greetingIndex, setGreetingIndex] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (messages.length > 0) return;
    const id = setInterval(() => {
      setGreetingIndex((i) => (i + 1) % GREETING_LINES.length);
    }, 3200);
    return () => clearInterval(id);
  }, [messages.length]);

  async function send(message: string) {
    const value = message.trim();
    if (!value || loading) return;
    setText("");
    setMessages((prev) => [...prev, { role: "user", text: value }]);
    setLoading(true);
    try {
      const res = await analyze(value);
      setMessages((prev) => [...prev, { role: "assistant", result: res }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Something went wrong";
      setMessages((prev) => [...prev, { role: "assistant-error", text: msg }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      {messages.length === 0 && (
        <div className="greeting-card">
          <div className="greeting-avatar-wrap">
            <span className="pulse-ring" />
            <span className="pulse-ring delay" />
            <div className="greeting-avatar">
              <Avatar size={104} />
            </div>
          </div>

          <div className="speech-bubble">
            <p className="greeting-line" key={greetingIndex}>
              {GREETING_LINES[greetingIndex]}
            </p>
            <p className="greeting-sub">
              Share what&apos;s on your mind. MindGuard gently classifies it and offers a supportive reply.
            </p>
          </div>

          <div className="suggestion-chips" style={{ justifyContent: "center", marginTop: 4 }}>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="chip"
                style={{ margin: 0, boxShadow: "none" }}
                onClick={() => send(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-thread">
        {messages.map((m, i) => {
          if (m.role === "user") {
            return (
              <div className="msg-row user" key={i}>
                <div className="msg-avatar">You</div>
                <div className="bubble">{m.text}</div>
              </div>
            );
          }
          if (m.role === "assistant-error") {
            return (
              <div className="msg-row assistant" key={i}>
                <div className="msg-avatar" style={{ background: "transparent" }}>
                  <Avatar size={34} />
                </div>
                <div className="bubble">
                  <div className="error" style={{ margin: 0 }}>{m.text}</div>
                </div>
              </div>
            );
          }

          const { result } = m;
          return (
            <div className="msg-row assistant" key={i}>
              <div className="msg-avatar" style={{ background: "transparent" }}>
                <Avatar size={34} />
              </div>
              <div className="bubble">
                <span className="pill">{result.prediction.label}</span>
                <span style={{ marginLeft: 10, color: "var(--muted)", fontSize: 12.5 }}>
                  {(result.prediction.confidence * 100).toFixed(1)}% confidence · {result.model_used}
                </span>

                <div style={{ marginTop: 14 }}>
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
                  <div style={{ marginTop: 16 }}>
                    <div style={{ color: "var(--muted)", fontSize: 12.5, marginBottom: 6 }}>
                      Empathetic response
                    </div>
                    <div style={{ lineHeight: 1.6 }}>{result.response}</div>
                  </div>
                )}

                <div className="disclaimer">{result.disclaimer}</div>
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="msg-row assistant">
            <div className="msg-avatar" style={{ background: "transparent" }}>
              <Avatar size={34} />
            </div>
            <div className="bubble">
              <span className="typing-dots">
                <span />
                <span />
                <span />
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(text);
            }
          }}
          placeholder="Type how you're feeling..."
          rows={1}
        />
        <button onClick={() => send(text)} disabled={loading || !text.trim()} aria-label="Send">
          <SendIcon />
        </button>
      </div>
    </div>
  );
}
