import React, { useEffect, useMemo, useRef, useState } from "react";
import "./CopilotWidget.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";

function nowTime() {
  return new Date().toLocaleTimeString();
}

export default function CopilotWidget() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState("IDLE");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi 👋 I’m your SOC Copilot. Ask about alerts, HIGH risk actions, DoS indicators, TCP states, etc.",
      ts: nowTime(),
    },
  ]);

  const [sourcesByMsg, setSourcesByMsg] = useState({}); // msgIndex -> sources[]
  const bottomRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && status !== "LOADING", [input, status]);

  // auto-scroll to bottom when open + messages change
  useEffect(() => {
    if (!open) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [open, messages.length]);

  const send = async () => {
    const q = input.trim();
    if (!q || !API_BASE) return;

    const userMsg = { role: "user", text: q, ts: nowTime() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setStatus("LOADING");

    try {
      const res = await fetch(`${API_BASE}/agent/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ query: q, top_k: 4 }),
      });

      // (ngrok sometimes returns HTML on errors, so parse safely)
      const raw = await res.text();
      let data;
      try {
        data = JSON.parse(raw);
      } catch {
        throw new Error("Non-JSON response from backend.");
      }

      const answer = data?.answer ?? "No answer returned.";
      const mode = data?.mode ?? "rag";

      const assistantMsg = {
        role: "assistant",
        text: answer,
        ts: nowTime(),
        meta: mode === "rag" ? `RAG • best_score=${(data.best_score ?? 0).toFixed(3)}` : "CHAT",
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // attach sources to the assistant message index
      const nextIndex = messages.length + 1; // user appended earlier, assistant will be next
      if (Array.isArray(data.sources) && data.sources.length > 0) {
        setSourcesByMsg((prev) => ({ ...prev, [nextIndex]: data.sources }));
      }
      setStatus("OK");
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${String(e.message || e)}`, ts: nowTime(), meta: "ERROR" },
      ]);
      setStatus("ERROR");
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) send();
    }
  };

  // Escape key closes widget
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <>
      {/* Floating launcher button */}
      <button
        className="copilot-launcher"
        onClick={() => setOpen((v) => !v)}
        aria-label="Open SOC Copilot"
        title="SOC Copilot"
      >
        {open ? "×" : "AI"}
      </button>

      {/* Popup widget */}
      {open && (
        <div className="copilot-widget" role="dialog" aria-label="SOC Copilot Assistant">
          <div className="copilot-header">
            <div className="copilot-title">
              <div className="copilot-title-main">SOC Copilot</div>
              <div className="copilot-title-sub">
                {API_BASE ? (
                  <>
                    Connected • <span className="mono">{API_BASE}</span>
                  </>
                ) : (
                  <>Set <span className="mono">VITE_API_BASE</span> in your .env</>
                )}
              </div>
            </div>

            <button className="copilot-close" onClick={() => setOpen(false)} aria-label="Close">
              Cancel
            </button>
          </div>

          <div className="copilot-body">
            {messages.map((m, idx) => (
              <div key={idx} className={`msg-row ${m.role}`}>
                <div className={`msg-bubble ${m.role}`}>
                  <div className="msg-text">{m.text}</div>
                  <div className="msg-meta">
                    <span className="msg-ts">{m.ts}</span>
                    {m.meta ? <span className="msg-tag">{m.meta}</span> : null}
                  </div>

                  {sourcesByMsg[idx] && (
                    <details className="sources">
                      <summary>Sources ({sourcesByMsg[idx].length})</summary>
                      <div className="sources-list">
                        {sourcesByMsg[idx].map((s, i) => (
                          <details key={i} className="source-item">
                            <summary>
                              [{i + 1}] {s.doc} #{s.chunk_index} • score {Number(s.score).toFixed(3)}
                            </summary>
                            <div className="source-text">{s.text}</div>
                          </details>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <div className="copilot-footer">
            <textarea
              className="copilot-input"
              placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={2}
            />
            <button className="copilot-send" onClick={send} disabled={!canSend}>
              {status === "LOADING" ? "..." : "Send"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
