"use client";

import { useEffect, useRef, useState } from "react";
import { BAND_LABEL, bandFor } from "@/lib/band";
import { nowLocal } from "@/lib/datetime";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Station = {
  category: string;
  state: string;
  state_mean_wait: number | null;
  queue_depth: number;
};

type PlanTask = { agent: string; label: string; delegated: boolean; reason: string };
type TraceStep = { phase: string; action: string; detail: string };

type Report = {
  summary: string;
  as_of: string;
  requested_as_of?: string;
  is_analog_replay?: boolean;
  replay_note?: string | null;
  clinic: string;
  category: string;
  plan: { objective: string; recognised: boolean; tasks: PlanTask[] };
  // What the supervisor did, in order, and its check of the result against the objective.
  trace: TraceStep[];
  // Set when the supervisor could not tell what the objective asks for: it ran
  // no agent and returns a question with the objectives it can handle.
  clarification?: {
    question: string;
    hint: string;
    options: { id: string; title: string; objective: string }[];
  } | null;
  goal: {
    achieved: boolean;
    met: number;
    total: number;
    criteria: { criterion: string; agent: string; met: boolean; detail: string }[];
  };
  consensus_wait_min: number | null;
  confidence: {
    confidence: number;
    confidence_label: string;
    checks: { check: string; passed: boolean; detail: string }[];
  };
  // Only the specialists the supervisor delegated to report findings.
  findings: {
    waiting_time?: {
      model_estimate_min: number;
      rag_median_min: number | null;
      rag_evidence_n: number;
      current_queue_depth: number;
    };
    bottlenecks?: { stations: Station[]; flagged: Station[] };
    forecast?: { next_hour_queue_depth: number[]; trend: string };
    resource_planning?: { options: { station: string; queue_depth: number; load_share_pct: number; option: string }[] };
    predictive_analysis?: {
      predicted_wait_min: number;
      predicted_wait_range_min: [number, number];
      outlook: string;
      risk_score: number | null;
      risk_label: string | null;
    };
  };
  recommendations: { priority: number; recommendation: string; impact: string }[];
  alerts: { level: string; message: string }[];
};

// The objectives a user can ask the supervisor for. The text is sent as-is;
// the supervisor decomposes it into the specialists to delegate to.
const OBJECTIVES = [
  {
    id: "bottlenecks",
    title: "Bottlenecks & resources",
    objective: "Analyse bottlenecks and suggest resource optimization",
  },
  {
    id: "waiting",
    title: "Waiting time & forecast",
    objective: "Estimate waiting time and forecast the queue",
  },
  {
    id: "full",
    title: "Full clinic review",
    objective: "Full clinic performance review",
  },
] as const;

const CUSTOM = "custom";

// The supervisor's own record of its run: one numbered step per action.
function SupervisorTrace({ trace }: { trace: TraceStep[] }) {
  return (
    <ol style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
      {trace.map((t, i) => (
        <li key={i} style={{ display: "grid", gridTemplateColumns: "22px 1fr", gap: 8, fontSize: 13 }}>
          <span
            style={{
              width: 22, height: 22, borderRadius: "50%", background: "var(--mediq-navy)",
              color: "#fff", fontSize: 11, fontWeight: 700, display: "inline-flex",
              alignItems: "center", justifyContent: "center",
            }}
          >
            {i + 1}
          </span>
          <span>
            <span className="station-reg" style={{ marginTop: 0, marginRight: 6 }}>{t.phase}</span>
            <strong>{t.action}</strong>
            <span className="muted" style={{ display: "block", marginTop: 2 }}>{t.detail}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

// One node of the supervisor graph, as streamed while the analysis runs.
type RunStep = {
  node: string;
  actor: string;
  label: string;
  status: "running" | "done";
  elapsed_s?: number;
  details: string[];
};

// Live record of the run under the Run button: each step appears when its agent
// starts and is ticked off when it reports back. Collapsible.
function ProcessWindow({
  steps, running, failed, open, onToggle,
}: {
  steps: RunStep[];
  running: boolean;
  failed: boolean;
  open: boolean;
  onToggle: (open: boolean) => void;
}) {
  const list = useRef<HTMLOListElement>(null);
  // keep the newest step in view while the list grows
  useEffect(() => {
    if (running && list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [steps, running]);
  const done = steps.filter((s) => s.status === "done").length;
  const total = steps.reduce((sum, s) => sum + (s.elapsed_s ?? 0), 0);
  const status = running
    ? `running · ${done} of ${steps.length} steps done`
    : failed
      ? `stopped after ${done} of ${steps.length} steps`
      : `finished · ${done} steps`;
  return (
    <details className="process" open={open} onToggle={(e) => onToggle(e.currentTarget.open)}>
      <summary>
        <span className="process-title">Process</span>
        <span className="process-status">{status}</span>
      </summary>
      <ol className="process-steps" aria-live="polite" ref={list}>
        {steps.map((s) => (
          <li key={s.node} className={`process-step ${s.status}`}>
            <span className="process-mark" aria-hidden="true">{s.status === "done" ? "✓" : ""}</span>
            <span>
              <span className="station-reg" style={{ marginTop: 0, marginRight: 6 }}>{s.actor}</span>
              <strong>{s.label}</strong>
              {s.status === "running" && <span className="muted"> — running…</span>}
              {s.details.map((d, i) => (
                <span key={i} className="muted" style={{ display: "block", marginTop: 2 }}>{d}</span>
              ))}
            </span>
            <span className="process-time">{!!s.elapsed_s && `${s.elapsed_s.toFixed(1)} s`}</span>
          </li>
        ))}
      </ol>
      {!running && !failed && steps.length > 0 && (
        <div className="process-foot">Agent time {total.toFixed(1)} s (specialists run in parallel)</div>
      )}
    </details>
  );
}

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
  const [objectiveId, setObjectiveId] = useState<string>(OBJECTIVES[0].id);
  const [customObjective, setCustomObjective] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [steps, setSteps] = useState<RunStep[]>([]);
  const [processOpen, setProcessOpen] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/v1/agent/dataset-range`)
      .then((r) => r.json())
      .then(setRange)
      .catch(() => {});
  }, []);

  const preset = OBJECTIVES.find((o) => o.id === objectiveId);
  const objectiveText = preset ? preset.objective : customObjective.trim();

  async function analyze(text: string = objectiveText) {
    setLoading(true);
    setError("");
    setSteps([]);
    setProcessOpen(true);
    try {
      const request = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clinic, category, as_of: `${asOf}:00`, objective: text }),
      };
      const res = await fetch(`${API}/api/v1/agent/analyze/stream`, request);
      if (res.status === 404) {
        // a backend started before the streaming endpoint existed: run without live steps
        const plain = await fetch(`${API}/api/v1/agent/analyze`, request);
        if (!plain.ok) throw new Error(`API ${plain.status}: ${await plain.text()}`);
        setReport(await plain.json());
        return;
      }
      if (!res.ok || !res.body) throw new Error(`API ${res.status}: ${await res.text()}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          if (!frame.startsWith("data: ")) continue;
          const ev = JSON.parse(frame.slice(6));
          if (ev.type === "step_start") {
            setSteps((all) => [
              ...all,
              { node: ev.node, actor: ev.actor, label: ev.label, status: "running", details: [] },
            ]);
          } else if (ev.type === "step_done") {
            setSteps((all) =>
              all.map((s) =>
                s.node === ev.node ? { ...s, status: "done", elapsed_s: ev.elapsed_s, details: ev.details } : s,
              ),
            );
          } else if (ev.type === "report") {
            setReport(ev.report);
          } else if (ev.type === "error") {
            throw new Error(ev.message);
          }
        }
      }
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
          <label id="objective-label">Objective for the supervisor</label>
          <div
            role="radiogroup"
            aria-labelledby="objective-label"
            style={{
              display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              border: "1px solid var(--mediq-navy)", borderRadius: "var(--r-sm)", overflow: "hidden",
              marginBottom: 6,
            }}
          >
            {[...OBJECTIVES, { id: CUSTOM, title: "Custom request" }].map((o, i) => {
              const selected = o.id === objectiveId;
              return (
                <button
                  key={o.id}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => setObjectiveId(o.id)}
                  style={{
                    padding: "10px 14px", border: "none", cursor: "pointer", fontSize: 13, fontWeight: 600,
                    borderLeft: i > 0 ? "1px solid var(--mediq-navy)" : "none",
                    background: selected ? "var(--mediq-navy)" : "var(--surface)",
                    color: selected ? "#fff" : "var(--mediq-navy)",
                  }}
                >
                  {o.title}
                </button>
              );
            })}
          </div>
          {preset ? (
            <p className="muted" style={{ marginBottom: 16 }}>
              Request sent to the supervisor: &ldquo;{preset.objective}&rdquo;
            </p>
          ) : (
            <div style={{ marginBottom: 16 }}>
              <input
                value={customObjective}
                onChange={(e) => setCustomObjective(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !loading && analyze()}
                placeholder="Type your own request, e.g. Where is the wait longest and do we need more staff?"
                aria-label="Custom request for the supervisor"
                maxLength={200}
                autoFocus
              />
              <p className="muted" style={{ marginTop: 4 }}>
                The supervisor works out which specialist agents your request needs, and asks you to
                clarify if it cannot tell.
              </p>
            </div>
          )}
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
            <button className="primary" style={{ gridColumn: "1 / -1" }} onClick={() => analyze()} disabled={loading}>
              {loading
                ? "Supervisor orchestrating specialist agents…"
                : "Run Multi-Agent Analysis"}
            </button>
          </div>
          {steps.length > 0 && (
            <ProcessWindow
              steps={steps}
              running={loading}
              failed={!!error}
              open={processOpen}
              onToggle={setProcessOpen}
            />
          )}
          {error && <p style={{ color: "var(--mediq-red)", marginTop: 10 }}>{error}</p>}
        </div>
      </div>

      {report?.clarification && (
        <div className="panel">
          <div className="panel-head">Supervisor — Clarification Needed</div>
          <div className="panel-body" aria-live="polite">
            <p className="qa-banner" style={{ marginBottom: 14 }}>
              No specialist agent was run. {report.clarification.question}
            </p>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
              {report.clarification.options.map((o) => (
                <button
                  key={o.id}
                  type="button"
                  className="secondary"
                  style={{ gridColumn: "auto", padding: "9px 16px" }}
                  disabled={loading}
                  title={o.objective}
                  onClick={() => {
                    setObjectiveId(o.id);
                    analyze(o.objective);
                  }}
                >
                  {o.title}
                </button>
              ))}
            </div>
            <p className="muted" style={{ marginBottom: 18 }}>{report.clarification.hint}</p>
            <div className="station-name">What the supervisor did</div>
            <SupervisorTrace trace={report.trace} />
          </div>
        </div>
      )}

      {report && !report.clarification && (
        <>
          <div className="panel">
            <div className="panel-head">Supervisor — Plan, Delegation &amp; Goal Validation</div>
            <div className="panel-body">
              <p style={{ marginBottom: 10, fontSize: 14 }}>
                Objective: <strong>&ldquo;{report.plan.objective}&rdquo;</strong>
              </p>
              <div className="station-name">Specialist agents</div>
              <div className="station-grid">
                {report.plan.tasks.map((t) => {
                  const result = report.goal.criteria.find((c) => c.agent === t.label);
                  return (
                  <div
                    key={t.agent}
                    className="station-card"
                    style={{
                      borderTopColor: t.delegated ? "var(--mediq-blue)" : "var(--border)",
                      opacity: t.delegated ? 1 : 0.6,
                    }}
                  >
                    <div className="station-name">{t.label}</div>
                    <div
                      style={{
                        fontSize: 13, fontWeight: 700, marginBottom: 4,
                        color: t.delegated ? "var(--mediq-blue)" : "var(--ink-mute)",
                      }}
                    >
                      {t.delegated ? "✓ Delegated" : "— Not delegated"}
                    </div>
                    <div className="muted" style={{ fontSize: 11 }}>{t.reason}</div>
                    {result && (
                      <div
                        style={{
                          marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border-soft)", fontSize: 12,
                          color: result.met ? "var(--ink)" : "var(--mediq-red)",
                        }}
                      >
                        <strong>{result.met ? "Reported back:" : "Could not deliver:"}</strong> {result.detail}
                      </div>
                    )}
                  </div>
                  );
                })}
              </div>

              <div
                style={{
                  display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
                  gap: 24, marginTop: 20,
                }}
              >
                <div>
                  <div className="station-name">What the supervisor did</div>
                  <SupervisorTrace trace={report.trace} />
                </div>
                <div>
                  <div className="station-name">Goal validation</div>
                  <div
                    style={{
                      fontSize: 15, fontWeight: 700, marginBottom: 8,
                      color: report.goal.achieved ? "var(--mediq-green)" : "var(--mediq-gold)",
                    }}
                  >
                    {report.goal.achieved ? "✓ Goal achieved" : "△ Goal partly achieved"} — {report.goal.met} of{" "}
                    {report.goal.total} criteria met
                  </div>
                  <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 6 }}>
                    {report.goal.criteria.map((c) => (
                      <li key={c.agent} style={{ fontSize: 13 }}>
                        <span style={{ color: c.met ? "var(--mediq-green)" : "var(--mediq-red)", fontWeight: 700 }}>
                          {c.met ? "✓" : "✗"}
                        </span>{" "}
                        {c.criterion}
                        <span className="muted" style={{ display: "block", marginLeft: 16 }}>
                          {c.agent} · {c.detail}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

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
                {report.consensus_wait_min != null && report.findings.waiting_time && (
                  <>
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
                      <div className="muted">LightGBM model</div>
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
                      <div className="muted">Queue now</div>
                      <div className="kpi-secondary">{report.findings.waiting_time.current_queue_depth}</div>
                    </div>
                  </>
                )}
                {report.findings.forecast && (
                  <div>
                    <div className="muted">Queue trend</div>
                    <div className="kpi-secondary">{report.findings.forecast.trend}</div>
                  </div>
                )}
                {report.findings.bottlenecks && (
                  <div>
                    <div className="muted">Bottlenecks flagged</div>
                    <div className="kpi-secondary">{report.findings.bottlenecks.flagged.length}</div>
                  </div>
                )}
                {report.findings.resource_planning && (
                  <div>
                    <div className="muted">Resource options</div>
                    <div className="kpi-secondary">{report.findings.resource_planning.options.length}</div>
                  </div>
                )}
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

          {report.findings.bottlenecks && (
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

          )}

          {report.findings.forecast && (
          <div className="panel">
            <div className="panel-head">LSTM Queue Forecast — Next Hour (15-min buckets)</div>
            <div className="panel-body">
              <div className="hourly-bars" style={{ maxWidth: 320 }}>
                {report.findings.forecast.next_hour_queue_depth.map((v, i, all) => {
                  const max = Math.max(...all, 1);
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

          )}

          {report.findings.predictive_analysis && (
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
                  {report.findings.predictive_analysis.risk_label ? (
                    <div
                      className="kpi-secondary"
                      style={{ color: RISK_COLOR[report.findings.predictive_analysis.risk_label], fontWeight: 700 }}
                    >
                      {report.findings.predictive_analysis.risk_label.toUpperCase()} (
                      {report.findings.predictive_analysis.risk_score})
                    </div>
                  ) : (
                    <div className="kpi-secondary" style={{ color: "var(--ink-mute)" }}>Not assessed</div>
                  )}
                </div>
                <div>
                  <div className="muted">Projected wait range (P10–P90)</div>
                  <div className="kpi-secondary">
                    {report.findings.predictive_analysis.predicted_wait_range_min[0].toFixed(0)}
                    –{report.findings.predictive_analysis.predicted_wait_range_min[1].toFixed(0)} min
                  </div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 8, fontSize: 11 }}>
                Combines the waiting-time and queue-forecast findings: the wait model&apos;s P10–P90 range,
                shifted by the queue trend, around a projected wait of{" "}
                {report.findings.predictive_analysis.predicted_wait_min.toFixed(0)} min.
                {!report.findings.predictive_analysis.risk_label &&
                  " The risk score also needs the bottleneck and resource-planning findings, which this objective did not request."}
              </div>
            </div>
          </div>
          )}

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
                  {report.recommendations.length === 0 && (
                    <tr>
                      <td colSpan={3} className="muted">No recommendations for this objective at this time.</td>
                    </tr>
                  )}
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
