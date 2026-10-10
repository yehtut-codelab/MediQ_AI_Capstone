"use client";

import { useEffect, useState } from "react";
import { BAND_LABEL, bandFor } from "@/lib/band";
import { nowLocal } from "@/lib/datetime";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Station = {
  category: string;
  state: string;
  state_mean_wait: number | null;
  queue_depth: number;
};

type Report = {
  summary: string;
  as_of: string;
  requested_as_of?: string;
  is_analog_replay?: boolean;
  replay_note?: string | null;
  clinic: string;
  category: string;
  consensus_wait_min: number;
  confidence: {
    confidence: number;
    confidence_label: string;
    checks: { check: string; passed: boolean; detail: string }[];
  };
  findings: {
    waiting_time: {
      model_estimate_min: number;
      rag_median_min: number | null;
      rag_evidence_n: number;
      current_queue_depth: number;
    };
    bottlenecks: { stations: Station[]; flagged: Station[] };
    forecast: { next_hour_queue_depth: number[]; trend: string };
    predictive_analysis: {
      predicted_wait_min: number;
      predicted_wait_range_min: [number, number];
      outlook: string;
      risk_score: number;
      risk_label: string;
    };
  };
  recommendations: { priority: number; recommendation: string; impact: string }[];
  alerts: { level: string; message: string }[];
};

const STATE_COLOR: Record<string, string> = {
  High: "var(--mediq-red)",
  Medium: "var(--mediq-gold)",
  Low: "var(--mediq-green)",
  unknown: "#999",
};

const RISK_COLOR: Record<string, string> = {
  high: "var(--mediq-red)",
  medium: "var(--mediq-gold)",
  low: "var(--mediq-green)",
};

export default function AgentConsole() {
  const [range, setRange] = useState<{ min: string; max: string; categories: string[] } | null>(null);
  const [clinic, setClinic] = useState("TTSH Eye Centre");
  const [category, setCategory] = useState("consultation");
  const [asOf, setAsOf] = useState(nowLocal);
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/v1/agent/dataset-range`)
      .then((r) => r.json())
      .then(setRange)
      .catch(() => {});
  }, []);

  async function analyze() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/v1/agent/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clinic, category, as_of: `${asOf}:00` }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      setReport(await res.json());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="panel">
        <div className="panel-head">
          AI Agent Console — Supervisor / Multi-Agent Analysis
        </div>
        <div className="panel-body">
          <div className="form-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div>
              <label>Clinic</label>
              <select value={clinic} onChange={(e) => setClinic(e.target.value)}>
                <option>TTSH Eye Centre</option>
                <option>Clinic 1A</option>
              </select>
            </div>
            <div>
              <label>Service Category</label>
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                {(range?.categories ?? ["consultation"]).map((c) => (
                  <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
                ))}
              </select>
            </div>
            <div>
              <label>
                Analysis time {range && `— trained-model data covers ${range.min.slice(0, 10)} … ${range.max.slice(0, 10)}`}
              </label>
              <input
                type="datetime-local"
                value={asOf}
                onChange={(e) => setAsOf(e.target.value)}
              />
            </div>
            <button className="primary" style={{ gridColumn: "1 / -1" }} onClick={analyze} disabled={loading}>
              {loading
                ? "Supervisor orchestrating specialist agents…"
                : "Run Multi-Agent Analysis"}
            </button>
          </div>
          {error && <p style={{ color: "var(--mediq-red)", marginTop: 10 }}>{error}</p>}
        </div>
      </div>

      {report && (
        <>
          {report.alerts.length > 0 && (
            <div className="panel">
              <div className="panel-head">Actionable Alerts</div>
              <div className="panel-body">
                {report.alerts.map((a, i) => (
                  <div key={i} className={`alert-row ${a.level === "critical" ? "crit" : "warn"}`}>
                    <strong>{a.level.toUpperCase()}</strong>&nbsp; {a.message}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="panel">
            <div className="panel-head">
              Fused Report · confidence {report.confidence.confidence_label} (
              {report.confidence.confidence})
            </div>
            <div className="panel-body" aria-live="polite">
              {report.is_analog_replay && (
                <p className="qa-banner" style={{ marginBottom: 12 }}>
                  {report.replay_note}
                </p>
              )}
              <div style={{ display: "flex", gap: 40, marginBottom: 12, flexWrap: "wrap" }}>
                <div>
                  <div className="muted">Consensus wait</div>
                  <div className={`estimate-value band-${bandFor(report.consensus_wait_min)}`}>
                    {Math.round(report.consensus_wait_min)}
                    <span style={{ fontSize: 18 }}> min</span>
                    <span className={`band-chip ${bandFor(report.consensus_wait_min)}`}>
                      {BAND_LABEL[bandFor(report.consensus_wait_min)]}
                    </span>
                  </div>
                </div>
                <div>
                  <div className="muted">XGBoost model</div>
                  <div className="kpi-secondary">
                    {report.findings.waiting_time.model_estimate_min} min
                  </div>
                </div>
                <div>
                  <div className="muted">RAG median ({report.findings.waiting_time.rag_evidence_n} events)</div>
                  <div className="kpi-secondary">
                    {report.findings.waiting_time.rag_median_min ?? "—"} min
                  </div>
                </div>
                <div>
                  <div className="muted">Queue now / trend</div>
                  <div className="kpi-secondary">
                    {report.findings.waiting_time.current_queue_depth} · {report.findings.forecast.trend}
                  </div>
                </div>
              </div>
              <p>{report.summary}</p>
              <div className="muted" style={{ marginTop: 8, fontSize: 11 }}>
                Validation checks:{" "}
                {report.confidence.checks.map((c) => (
                  <span key={c.check} style={{ marginRight: 12 }}>
                    {c.passed ? "✓" : "✗"} {c.check} ({c.detail})
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">Bottleneck Detection — HMM Congestion States</div>
            <div className="panel-body">
              <div className="station-grid">
                {report.findings.bottlenecks.stations
                  .filter((s) => s.state !== "unknown" || s.queue_depth > 0)
                  .map((s) => (
                    <div
                      key={s.category}
                      className="station-card"
                      style={{ borderTopColor: STATE_COLOR[s.state] }}
                    >
                      <div className="station-name">{s.category.replace(/_/g, " ")}</div>
                      <div className="station-median" style={{ color: STATE_COLOR[s.state], fontSize: 20 }}>
                        {s.state}
                      </div>
                      <div className="muted" style={{ fontSize: 11 }}>
                        {s.queue_depth} in queue
                        {s.state_mean_wait != null && ` · state mean ${s.state_mean_wait}m`}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">LSTM Queue Forecast — Next Hour (15-min buckets)</div>
            <div className="panel-body">
              <div className="hourly-bars" style={{ maxWidth: 320 }}>
                {report.findings.forecast.next_hour_queue_depth.map((v, i) => {
                  const max = Math.max(...report.findings.forecast.next_hour_queue_depth, 1);
                  return (
                    <div
                      key={i}
                      className="hourly-bar"
                      role="img"
                      aria-label={`+${(i + 1) * 15} minutes: ${v} patients in queue`}
                      title={`+${(i + 1) * 15} min: ${v}`}
                      style={{
                        height: `${Math.max(4, (v / max) * 40)}px`,
                        background: "var(--mediq-blue)",
                      }}
                    />
                  );
                })}
              </div>
              <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                {report.findings.forecast.next_hour_queue_depth.map((v, i) => `+${(i + 1) * 15}m: ${v}`).join(" · ")}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">Predictive Analytics Agent — Synthesis</div>
            <div className="panel-body">
              <div style={{ display: "flex", gap: 40, flexWrap: "wrap" }}>
                <div>
                  <div className="muted">Outlook</div>
                  <div className="kpi-secondary" style={{ textTransform: "capitalize" }}>
                    {report.findings.predictive_analysis.outlook}
                  </div>
                </div>
                <div>
                  <div className="muted">Operational risk</div>
                  <div
                    className="kpi-secondary"
                    style={{ color: RISK_COLOR[report.findings.predictive_analysis.risk_label], fontWeight: 700 }}
                  >
                    {report.findings.predictive_analysis.risk_label.toUpperCase()} (
                    {report.findings.predictive_analysis.risk_score})
                  </div>
                </div>
                <div>
                  <div className="muted">Projected wait (next cycle)</div>
                  <div className="kpi-secondary">
                    {report.findings.predictive_analysis.predicted_wait_range_min[0].toFixed(0)}
                    –{report.findings.predictive_analysis.predicted_wait_range_min[1].toFixed(0)} min
                  </div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 8, fontSize: 11 }}>
                Synthesizes waiting-time, bottleneck, queue-depth-forecast, and resource-planning
                findings — trend-adjusted from the point estimate of{" "}
                {report.findings.predictive_analysis.predicted_wait_min.toFixed(0)} min.
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">Recommendations (ranked)</div>
            <div className="panel-body">
              <table className="evidence-table" style={{ marginTop: 0 }}>
                <thead>
                  <tr>
                    <th>P</th>
                    <th>Recommendation</th>
                    <th>Expected impact</th>
                  </tr>
                </thead>
                <tbody>
                  {report.recommendations.map((r, i) => (
                    <tr key={i}>
                      <td style={{ fontWeight: 700, color: r.priority === 1 ? "var(--mediq-red)" : r.priority === 2 ? "var(--mediq-gold)" : "inherit" }}>
                        {r.priority}
                      </td>
                      <td>{r.recommendation}</td>
                      <td className="muted">{r.impact}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}
