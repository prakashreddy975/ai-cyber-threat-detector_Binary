import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import CopilotWidget from "./components/CopilotWidget";

import { Line, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, ArcElement, Tooltip, Legend, Filler);

async function safeFetchJson(url, options = {}) {
  const res = await fetch(url, {
    cache: "no-store",
    headers: {
      "ngrok-skip-browser-warning": "true",
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await res.text();

  try {
    return {
      ok: res.ok,
      status: res.status,
      json: JSON.parse(text),
      raw: null,
    };
  } catch {
    return {
      ok: res.ok,
      status: res.status,
      json: null,
      raw: text,
    };
  }
}

const MetricCard = ({ label, value, sub, color }) => (
  <div className="hud-card" style={{ borderLeftColor: color }}>
    <div className="hud-label">{label}</div>
    <div className="hud-value" style={{ color }}>
      {value}
    </div>
    <div className="hud-sub">{sub}</div>
  </div>
);

const Pill = ({ text, kind }) => <span className={`pill ${kind}`}>{text}</span>;

export default function App() {
  // ---- Config (env first, fallback local) ----
  const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
  const WS_URL = import.meta.env.VITE_WS_URL || "";
  console.log("ENV VITE_API_BASE =", import.meta.env.VITE_API_BASE);
  console.log("ENV VITE_WS_URL   =", import.meta.env.VITE_WS_URL);
  console.log("USING API_BASE    =", API_BASE);
  console.log("USING WS_URL      =", WS_URL);

  // ---- WS state ----
  const [logs, setLogs] = useState([]);
  const [latencyData, setLatencyData] = useState(new Array(20).fill(0));
  const [riskCounts, setRiskCounts] = useState({ LOW: 0, MEDIUM: 0, HIGH: 0 });
  const [wsStatus, setWsStatus] = useState("DISCONNECTED");

  // ---- REST state ----
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const [alertsOpen, setAlertsOpen] = useState(false);
  const [alerts, setAlerts] = useState([]);

  const [schemaOpen, setSchemaOpen] = useState(false);
  const [expectedCols, setExpectedCols] = useState(null);

  const [copilotQ, setCopilotQ] = useState("");
  const [copilotA, setCopilotA] = useState("");
  const [copilotSources, setCopilotSources] = useState([]);
  const [copilotLoading, setCopilotLoading] = useState(false);


  // ---- Quick Predict ----
  const [formJson, setFormJson] = useState(`{
  "features": {
    "id": 1,
    "dur": 0.2,
    "proto": "tcp",
    "service": "http",
    "state": "FIN",
    "spkts": 4,
    "dpkts": 2,
    "sbytes": 200,
    "dbytes": 100
  }
}`);
  const [predictResp, setPredictResp] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);

  // ----------------------------
  // WebSocket connect
  // ----------------------------
  useEffect(() => {
    if (!WS_URL) return;
  
    let socket = null;
    let isUnmounting = false;
  
    // Delay connect 1 tick so StrictMode cleanup doesn’t kill it mid-handshake
    const t = setTimeout(() => {
      if (isUnmounting) return;
  
      socket = new WebSocket(WS_URL);
  
      socket.onopen = () => setWsStatus("CONNECTED");
      socket.onclose = () => setWsStatus("DISCONNECTED");
      socket.onerror = () => setWsStatus("ERROR");
  
      socket.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        if (data.type === "ping" || data.type === "hello") return;
  
        setLogs((prev) => [data, ...prev].slice(0, 12));
  
        const latency = Number(data.latency ?? 0);
        setLatencyData((prev) => [...prev, latency].slice(-20));
  
        const rl = String(data.risk_level || "LOW").toUpperCase();
        setRiskCounts((prev) => ({
          ...prev,
          [rl in prev ? rl : "LOW"]: prev[rl in prev ? rl : "LOW"] + 1,
        }));
      };
    }, 150);
  
    return () => {
      isUnmounting = true;
      clearTimeout(t);
  
      // Only close if actually opened or connecting safely
      try {
        if (socket && socket.readyState === WebSocket.OPEN) socket.close(1000, "unmount");
        else if (socket && socket.readyState === WebSocket.CONNECTING) socket.close();
      } catch {}
    };
  }, [WS_URL]);

  // ----------------------------
  // Poll /health + /metrics every 5s
  // ----------------------------
  useEffect(() => {
    let alive = true;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
      
        const text = await res.text(); // read raw body first
        let body;
      
        try {
          body = JSON.parse(text);
        } catch {
          // If ngrok returns HTML or anything non-JSON
          body = { status: res.ok ? "running" : "down", raw: text.slice(0, 120) };
        }
      
        console.log("HEALTH:", res.status, body);
      
        if (!alive) return;
        setHealth({ ...body, _httpStatus: res.status, _ok: res.ok });
      
      } catch (e) {
        console.log("HEALTH FETCH FAILED:", e);
        if (!alive) return;
        setHealth(null);
      }
    };

    poll();
    const t = setInterval(poll, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [API_BASE]);


  useEffect(() => {
    let alive = true;
  
    async function loadMetrics() {
      const r = await safeFetchJson(`${API_BASE}/metrics`);
      console.log("METRICS:", r);
  
      if (!alive) return;
      if (r.json) setMetrics(r.json);
    }
  
    loadMetrics();
  }, [API_BASE]);


  // ----------------------------
  // Load alerts when drawer open + refresh every 3s while open
  // ----------------------------
  useEffect(() => {
    if (!alertsOpen) return; // ⛔ don’t fetch if drawer closed
  
    let alive = true;
  
    async function loadAlerts() {
      try {
        const r = await safeFetchJson(`${API_BASE}/alerts?limit=50`);
        console.log("ALERTS:", r);
  
        if (!alive) return;
  
        if (r.json?.alerts) {
          setAlerts(r.json.alerts);
        }
      } catch (e) {
        console.error("Failed to load alerts:", e);
      }
    }
  
    loadAlerts();
  
    return () => {
      alive = false;
    };
  }, [API_BASE, alertsOpen]); // ✅ added alertsOpen
  


  // ----------------------------
  // Load expected columns when schema drawer open
  // ----------------------------
  useEffect(() => {
    async function loadColumns() {
      const r = await safeFetchJson(`${API_BASE}/expected_columns`);
      console.log("COLUMNS:", r);
  
      if (r.json?.available) {
        setExpectedCols(r.json.columns);
      }
    }
  
    loadColumns();
  }, [API_BASE]);


  async function askCopilot() {
    if (!copilotQ.trim()) return;
  
    setCopilotLoading(true);
    setCopilotA("");
    setCopilotSources([]);
  
    try {
      const res = await fetch(`${API_BASE}/agent/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ query: copilotQ, top_k: 4}),
      });
  
      const text = await res.text();
      const json = JSON.parse(text);
  
      setCopilotA(json.answer || "");
      setCopilotSources(json.sources || []);
    } catch (e) {
      setCopilotA("Copilot error: " + String(e));
    } finally {
      setCopilotLoading(false);
    }
  }
  
  
  // ----------------------------
  // Quick Predict: POST /predict
  // ----------------------------
  const doPredict = async () => {
    setPredictLoading(true);
    setPredictResp(null);

    try {
      const body = JSON.parse(formJson);

      const r = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const j = await r.json();
      setPredictResp({ ok: r.ok, status: r.status, body: j });
    } catch (e) {
      setPredictResp({ ok: false, status: 0, body: { error: String(e) } });
    } finally {
      setPredictLoading(false);
    }
  };

  // ----------------------------
  // Derived metrics (your old HUD logic)
  // ----------------------------
  const latest = logs[0];

  const systemStatus = useMemo(() => {
    if (!latest) return "STABLE";
    return latest.prediction === 1 && String(latest.risk_level).toUpperCase() === "HIGH"
      ? "CRITICAL"
      : "STABLE";
  }, [latest]);

  const avgThreatScore = useMemo(() => {
    if (logs.length === 0) return 0;
    const s = logs.reduce((acc, x) => acc + Number(x.threat_score || 0), 0);
    return s / logs.length;
  }, [logs]);

  const attackCount = useMemo(() => logs.filter((x) => x.prediction === 1).length, [logs]);
  const latestLatency = useMemo(() => Number(latest?.latency ?? 0), [latest]);

  const colors = { LOW: "#6ac174", MEDIUM: "#ff9830", HIGH: "#f2495c" };

  const severityClass = (riskLevel) => {
    const r = String(riskLevel || "").toUpperCase();
    if (r === "HIGH") return "severity-high";
    if (r === "MEDIUM") return "severity-med";
    return "";
  };

  const backendUp =
  !!health && (
    String(health.status || "").toLowerCase() === "running" ||
    String(health.status || "").toLowerCase() === "ok" ||
    String(health.status || "").toLowerCase() === "healthy" ||
    health.status === true
  );

  return (
    <div className="dashboard-wrapper">
      {/* Header */}
      <div className="topbar">
        <div>
          <div className="title">AI Threat Detector</div>
          <div className="subtitle">
            WebSocket: <span className={`ws-pill ${wsStatus.toLowerCase()}`}>{wsStatus}</span>
            <span className="dot">•</span>
            Backend: <span className={`ws-pill ${backendUp ? "connected" : "error"}`}>{backendUp ? "UP" : "DOWN"}</span>
            <span className="dot">•</span>
            Source: <span className="mono">{WS_URL || "Set VITE_WS_URL in .env"}</span>
          </div>
        </div>

        {/* NEW: top actions */}
        <div className="top-actions">
          <button className="chip" onClick={() => setAlertsOpen(true)}>
            Alerts ({metrics?.stored_alerts ?? 0})
          </button>
          <button className="chip" onClick={() => setSchemaOpen(true)}>
            Schema
          </button>
        </div>
      </div>

      {/* 1) EXEC HUD METRICS (old) */}
      <div className="metrics-row">
        <MetricCard
          label="System Status"
          value={systemStatus}
          sub="Realtime inference state"
          color={systemStatus === "CRITICAL" ? colors.HIGH : colors.LOW}
        />
        <MetricCard
          label="Avg Threat Score"
          value={avgThreatScore.toFixed(3)}
          sub="Mean score (last 12 events)"
          color="#5794f2"
        />
        <MetricCard
          label="Attack Count"
          value={attackCount}
          sub="Attacks in last 12 events"
          color="#b877d9"
        />
        <MetricCard
          label="Latest Latency"
          value={`${latestLatency.toFixed(2)} ms`}
          sub="Model inference latency"
          color="#ff9830"
        />
      </div>

      {/* NEW: Backend Diagnostics (/metrics) */}
      <div className="status-panel">
        <div className="status-title">Backend Diagnostics (GET /metrics)</div>
        <div className="status-grid">
          <div className="status-item">
            <div className="status-k">stored_alerts</div>
            <div className="status-v">{metrics?.stored_alerts ?? "-"}</div>
          </div>
          <div className="status-item">
            <div className="status-k">threshold</div>
            <div className="status-v">{metrics?.threshold ?? "-"}</div>
          </div>
          <div className="status-item">
            <div className="status-k">ws_clients</div>
            <div className="status-v">{metrics?.ws_clients ?? "-"}</div>
          </div>
          <div className="status-item">
            <div className="status-k">expected_cols</div>
            <div className="status-v">{metrics?.n_expected_cols ?? "-"}</div>
          </div>
        </div>
      </div>

      {/* 2) CHARTS ROW (old) */}
      <div className="charts-row">
        {/* Pulse Chart */}
        <div className="panel">
          <div className="panel-header">Inference Pulse (Latency)</div>
          <div className="panel-body-fixed">
            <Line
              data={{
                labels: new Array(20).fill(""),
                datasets: [
                  {
                    data: latencyData,
                    borderColor: "#5794f2",
                    fill: true,
                    backgroundColor: "rgba(87, 148, 242, 0.08)",
                    tension: 0.15,
                    pointRadius: 0,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                  x: { display: false },
                  y: { grid: { color: "#2c3235" } },
                },
                plugins: { legend: { display: false } },
              }}
            />
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="panel panel-right">
          <div className="panel-header panel-header-full">Risk Distribution</div>

          <div className="donut-wrap">
            <Doughnut
              data={{
                labels: Object.keys(riskCounts),
                datasets: [
                  {
                    data: Object.values(riskCounts),
                    backgroundColor: Object.keys(riskCounts).map((k) => colors[k] || "#515151"),
                    borderWidth: 0,
                    cutout: "80%",
                  },
                ],
              }}
              options={{ maintainAspectRatio: false, plugins: { legend: { display: false } } }}
            />
          </div>

          <div className="legend-grid">
            {Object.entries(riskCounts).map(([label, val]) => (
              <div key={label} className="legend-item" style={{ borderLeftColor: colors[label] }}>
                <span className="legend-label">{label}</span>
                <span className="legend-value">{val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* NEW: Quick Predict panel (/predict) */}
      <div className="panel">
        <div className="panel-header">Quick Predict (POST /predict)</div>
        <div className="quick-predict">
          <textarea
            className="jsonbox"
            value={formJson}
            onChange={(e) => setFormJson(e.target.value)}
            spellCheck={false}
          />

          <div className="qp-actions">
            <button className="chip" onClick={doPredict} disabled={predictLoading}>
              {predictLoading ? "Running..." : "Run Predict"}
            </button>

            {predictResp?.ok ? (
              <Pill kind="ok" text="OK" />
            ) : predictResp ? (
              <Pill kind="bad" text={`ERR ${predictResp.status}`} />
            ) : null}

            <div className="hint mono">POST {API_BASE}/predict</div>
          </div>

          {predictResp ? (
            <pre className="prebox">{JSON.stringify(predictResp.body, null, 2)}</pre>
          ) : (
            <div className="hint">
              Tip: paste <span className="mono">{`{ "features": { ... } }`}</span> and click Run.
            </div>
          )}
        </div>
      </div>

      {/* 3) INCIDENT LOGS TABLE (old) */}
      <div className="log-table-container">
        <div className="log-header">LIVE_INFERENCE_FEED</div>

        <table className="grafana-table">
          <thead>
            <tr>
              <th>TIMESTAMP</th>
              <th>PREDICTION</th>
              <th>RISK</th>
              <th>THREAT SCORE</th>
              <th>LATENCY</th>
              <th className="th-right">ACTION</th>
            </tr>
          </thead>

          <tbody>
            {logs.map((log, i) => {
              const risk = String(log.risk_level || "LOW").toUpperCase();
              const isAttack = log.prediction === 1;
              const score = Number(log.threat_score ?? 0);
              const latency = Number(log.latency ?? 0);

              return (
                <tr key={i} className={severityClass(risk)}>
                  <td className="td-muted">{new Date().toLocaleTimeString()}</td>
                  <td className="td-strong" style={{ color: isAttack ? colors.HIGH : colors.LOW }}>
                    {isAttack ? "ATTACK (1)" : "NORMAL (0)"}
                  </td>
                  <td className="td-strong">{risk}</td>
                  <td>
                    <div className="conf-row">
                      <div className="conf-bar">
                        <div
                          className="conf-fill"
                          style={{
                            width: `${Math.round(score * 100)}%`,
                            background: risk === "HIGH" ? colors.HIGH : risk === "MEDIUM" ? colors.MEDIUM : colors.LOW,
                          }}
                        />
                      </div>
                      <span className="conf-text">{score.toFixed(4)}</span>
                    </div>
                  </td>
                  <td className="td-muted">{latency.toFixed(2)} ms</td>
                  <td className="th-right">
                    <button className="action-btn">{risk === "HIGH" ? "ISOLATE" : "IGNORE"}</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* NEW: Alerts Drawer (/alerts) */}
      {alertsOpen ? (
        <div className="drawer-backdrop" onClick={() => setAlertsOpen(false)}>
          <div className="drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <div className="drawer-title">Stored Alerts (GET /alerts)</div>
              <button className="chip" onClick={() => setAlertsOpen(false)}>
                Close
              </button>
            </div>

            <div className="drawer-body">
              {alerts.length === 0 ? (
                <div className="hint">No alerts stored yet (or server restarted).</div>
              ) : (
                <div className="alerts-list">
                  {alerts.map((a, idx) => (
                    <div key={idx} className={`alert-card ${severityClass(a.risk_level)}`}>
                      <div className="alert-top">
                        <div className="mono">{a.timestamp}</div>
                        <Pill
                          kind={String(a.risk_level).toUpperCase() === "HIGH" ? "bad" : "warn"}
                          text={String(a.risk_level).toUpperCase()}
                        />
                      </div>

                      <div className="alert-mid">
                        <div>
                          <b>prediction:</b> {a.prediction}
                        </div>
                        <div>
                          <b>score:</b> {Number(a.threat_score ?? 0).toFixed(4)}
                        </div>
                        <div>
                          <b>latency:</b> {a.latency} ms
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {/* NEW: Schema Drawer (/expected_columns) */}
      {schemaOpen ? (
        <div className="drawer-backdrop" onClick={() => setSchemaOpen(false)}>
          <div className="drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <div className="drawer-title">Expected Columns (GET /expected_columns)</div>
              <button className="chip" onClick={() => setSchemaOpen(false)}>
                Close
              </button>
            </div>

            <div className="drawer-body">
              {!expectedCols ? (
                <div className="hint">Loading schema…</div>
              ) : expectedCols.available ? (
                <>
                  <div className="hint">
                    Detected <b>{expectedCols.n_cols}</b> columns from backend preprocessing.
                  </div>
                  <pre className="prebox">{JSON.stringify(expectedCols.columns, null, 2)}</pre>
                </>
              ) : (
                <div className="hint">Backend did not report expected columns.</div>
              )}
            </div>
          </div>
        </div>
      ) : null}
      <CopilotWidget />
    </div>
  );
}
