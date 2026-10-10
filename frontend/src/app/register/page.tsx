"use client";

import { useEffect, useState } from "react";
import SuggestField from "@/components/SuggestField";
import { BAND_LABEL } from "@/lib/band";
import { CURRENT_ISSUE_OPTIONS, DIAGNOSIS_OPTIONS, HISTORY_OPTIONS } from "@/lib/clinicalOptions";
import { nowLocal, todayAt } from "@/lib/datetime";
import { GoldenSamples, type GoldenSample } from "@/components/GoldenSamples";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const CLINICS = ["TTSH Eye Centre", "Clinic 1A"];
const SERVICE_TYPES = [
  "Consultation", "OCT", "VA", "HVF", "Orthoptic", "Biometry", "PAT",
  "Diagnostic", "Financial Con", "Pre Consultation Test",
];

type PathwayStep = {
  station: string;
  eta_offset_min: number;
  expected_wait_min: number;
  expected_service_min: number;
  note: string | null;
};

type ForecastBucket = { minutes_ahead: number; queue_depth: number };

type RegisteredPatient = {
  patient_id: number;
  display_name: string | null;
  clinic: string;
  service_type: string;
  arrival_datetime: string;
  appointment_datetime: string | null;
  queue_position: number;
  priority: number;
  priority_label: string;
  triage_reasons: string[];
  pathway_label: string | null;
  pathway: PathwayStep[];
  total_visit_min: number | null;
  estimated_wait_min: number;
  range_min: [number, number];
  status_band: "green" | "amber" | "red";
  explanation: string;
  registered_at: string;
  rag_estimate_min?: number | null;
  rag_range_min?: [number, number] | null;
  rag_evidence_n?: number | null;
  model_category: string | null;
  model_estimate_min: number | null;
  model_mae_min: number | null;
  // Breach = predicted to wait past the clinical limit for the service. Only on a fresh registration.
  model_breach_limit_min?: number | null;
  model_breach_probability?: number | null;
  model_breach_flag?: boolean | null;
  model_queue_depth: number | null;
  model_congestion_state: "Low" | "Medium" | "High" | "unknown" | null;
  model_congestion_mean_wait_min: number | null;
  model_forecast: ForecastBucket[];
  model_forecast_trend: "rising" | "falling" | "stable" | "no-data" | null;
  model_replayed_at: string | null;
  model_is_analog_replay: boolean;
  model_replay_note: string | null;
  model_unavailable_reason: string | null;
};

type Metrics = { MAE: number; RMSE: number; R2: number };
type DataWindow = { events: number; from: string; to: string };

type ModelCard = {
  name: string;
  algorithm: string;
  features: string[];
  // Inputs this app has no value for; the station's typical value is used.
  defaulted_features: string[];
  train: DataWindow;
  test: DataWindow;
  // Hold-out accuracy: with every input as trained, as this app calls the
  // model, and as this app calls it when no appointment time is entered.
  evaluation: { as_trained: Metrics; app: Metrics; app_no_appointment: Metrics };
  range: { share_within_p10_p90: number };
  breach: { precision: number; recall: number };
  baselines: ({ label: string } & Metrics)[];
};

const FEATURE_LABEL: Record<string, string> = {
  service_category: "Service category",
  clinic_encoded: "Clinic",
  hour_of_day: "Hour of day",
  day_of_week: "Day of week",
  month: "Month",
  visit_position: "Step in the visit",
  appointment_lag_clipped: "Early / late vs appointment",
  queue_depth: "Patients already waiting",
  queue_depth_lag1: "Queue 15 min ago",
  queue_drain_rate: "How fast the queue is moving",
  room_queue_depth: "Queue for the room",
  cross_cat_load: "Patients waiting, all services",
  consultation_queue_at_arrival: "Consultation queue",
  appointment_density_next30: "Appointments in the next 30 min",
  room_number_encoded: "Room",
  prior_visit_count: "Previous visits",
  days_to_next_holiday: "Days to next holiday",
  days_since_last_holiday: "Days since last holiday",
};

const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-SG", { day: "numeric", month: "short", year: "numeric" });

// Hold-out error of the model, as this app calls it, against simple
// baselines: one bar per predictor, shorter is better.
function ModelStats({ card }: { card: ModelCard }) {
  const ev = card.evaluation.app;
  const rows = [
    ...card.baselines.map((b) => ({ label: b.label + " (baseline)", mae: b.MAE, model: false })),
    { label: "LightGBM model", mae: ev.MAE, model: true },
  ];
  const max = Math.max(...rows.map((r) => r.mae));
  const best = Math.min(...card.baselines.map((b) => b.MAE));
  const gain = Math.round(((best - ev.MAE) / best) * 100);
  return (
    <>
      <p className="muted" style={{ marginTop: 0, marginBottom: 14, fontSize: 13 }}>
        Every wait on this page is predicted by a machine-learning model ({card.algorithm}) trained on{" "}
        <strong>{card.train.events.toLocaleString()}</strong> historical visit events from{" "}
        {fmtDate(card.train.from)} to {fmtDate(card.train.to)}. The figures below were measured on{" "}
        {card.test.events.toLocaleString()} later events ({fmtDate(card.test.from)} to {fmtDate(card.test.to)}),
        with the model given only what this page knows at registration.
      </p>
      <div style={{ display: "flex", gap: 36, flexWrap: "wrap", marginBottom: 18 }}>
        <div>
          <div className="muted">Typical error (test MAE)</div>
          <div className="kpi-primary" style={{ color: "var(--mediq-blue)" }}>
            ±{ev.MAE.toFixed(1)}
            <span style={{ fontSize: 18 }}> min</span>
          </div>
          <div className="muted">
            ±{card.evaluation.app_no_appointment.MAE.toFixed(1)} min with no appointment time
          </div>
        </div>
        <div>
          <div className="muted">RMSE</div>
          <div className="kpi-secondary">{ev.RMSE.toFixed(1)} min</div>
        </div>
        <div>
          <div className="muted">R² (variation explained)</div>
          <div className="kpi-secondary">{ev.R2.toFixed(2)}</div>
        </div>
        <div>
          <div className="muted">vs best baseline</div>
          <div className="kpi-secondary" style={{ color: "var(--mediq-green)" }}>{gain}% lower error</div>
        </div>
        <div>
          <div className="muted">If the room were known</div>
          <div className="kpi-secondary">±{card.evaluation.as_trained.MAE.toFixed(1)} min</div>
        </div>
      </div>

      <div className="muted" style={{ marginBottom: 6 }}>Mean absolute error on test data — lower is better</div>
      <div style={{ display: "grid", gap: 6, maxWidth: 560, marginBottom: 16 }}>
        {rows.map((r) => (
          <div key={r.label} style={{ display: "grid", gridTemplateColumns: "170px 1fr 70px", gap: 10, alignItems: "center" }}>
            <span style={{ fontSize: 12.5, fontWeight: r.model ? 700 : 500 }}>{r.label}</span>
            <div style={{ background: "var(--border-soft)", borderRadius: 4, height: 14 }}>
              <div
                style={{
                  width: `${(r.mae / max) * 100}%`,
                  height: "100%",
                  borderRadius: 4,
                  background: r.model ? "var(--mediq-blue)" : "#9aa3ad",
                }}
              />
            </div>
            <span style={{ fontSize: 12.5, fontWeight: r.model ? 700 : 500 }}>{r.mae.toFixed(1)} min</span>
          </div>
        ))}
      </div>

      <div className="muted" style={{ marginBottom: 6 }}>
        Inputs the model uses — grey inputs are not known at registration, so the station&apos;s typical value is used
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
        {card.features.map((f) => (
          <span
            key={f}
            className="station-reg"
            style={{
              marginTop: 0, fontSize: 11, padding: "3px 9px",
              ...(card.defaulted_features.includes(f) && { background: "var(--border-soft)", color: "var(--ink-mute)" }),
            }}
          >
            {FEATURE_LABEL[f] ?? f}
          </span>
        ))}
      </div>
      <p className="muted">
        Likely range: {Math.round(card.range.share_within_p10_p90 * 100)}% of test waits fell inside the
        range shown. Limit warning: {Math.round(card.breach.precision * 100)}% of flagged patients did wait
        past the limit, and {Math.round(card.breach.recall * 100)}% of patients who did were flagged.
      </p>
    </>
  );
}

// A registration has no id in the API; patient id + registration instant
// identifies one row (the same patient can be registered more than once).
const rowKey = (p: RegisteredPatient) => `${p.patient_id}|${Date.parse(p.registered_at)}`;

const NEW_ROW_ID = "new-patient-row";

const ordinal = (n: number) => {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
};

const PRIORITY_COLOR: Record<number, string> = {
  1: "var(--mediq-red, #c0392b)",
  2: "#b9770e",
  3: "#1e8449",
};

const CONGESTION_COLOR: Record<string, string> = {
  High: "var(--mediq-red)",
  Medium: "var(--mediq-gold)",
  Low: "var(--mediq-green)",
  unknown: "#999",
};

const TREND_ARROW: Record<string, string> = {
  rising: "↑ rising",
  falling: "↓ falling",
  stable: "→ stable",
  "no-data": "— no data",
};

function PriorityBadge({ p, label }: { p: number; label: string }) {
  return (
    <span
      style={{
        background: PRIORITY_COLOR[p] ?? PRIORITY_COLOR[3],
        color: "#fff",
        borderRadius: 4,
        padding: "2px 8px",
        fontSize: 12,
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

export default function RegisterPatient() {
  const [name, setName] = useState("");
  const [patientId, setPatientId] = useState("");
  const [clinic, setClinic] = useState(CLINICS[0]);
  const [serviceType, setServiceType] = useState(SERVICE_TYPES[0]);
  const [arrival, setArrival] = useState(nowLocal);
  const [appointment, setAppointment] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [history, setHistory] = useState("");
  const [currentIssue, setCurrentIssue] = useState("");
  // A picked golden sample pins the clinic state; editing clinic, service or arrival unpins it.
  const [golden, setGolden] = useState<GoldenSample | null>(null);
  const [registered, setRegistered] = useState<RegisteredPatient | null>(null);
  const [queue, setQueue] = useState<RegisteredPatient[]>([]);
  const [modelCard, setModelCard] = useState<ModelCard | null>(null);
  const [modelOpen, setModelOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refreshQueue() {
    try {
      const res = await fetch(`${API}/api/v1/patients`);
      if (res.ok) setQueue(await res.json());
    } catch {
      /* queue list is non-critical */
    }
  }

  useEffect(() => {
    refreshQueue();
    fetch(`${API}/api/v1/models/wait-time`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setModelCard)
      .catch(() => {});
  }, []);

  function pickGolden(s: GoldenSample) {
    setName(s.display_name);
    setPatientId("");
    setClinic(s.clinic);
    setServiceType(s.service_type);
    setArrival(todayAt(s.arrival_time));
    setAppointment(todayAt(s.appointment_time));
    setDiagnosis(s.diagnosis);
    setHistory(s.medical_history);
    setCurrentIssue(s.current_issue);
    setGolden(s);
    setError("");
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/v1/patients/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId ? Number(patientId) : null,
          display_name: name || null,
          clinic,
          service_type: serviceType,
          arrival_datetime: `${arrival}:00+08:00`,
          appointment_datetime: appointment ? `${appointment}:00+08:00` : null,
          replay_at: golden?.replay_at ?? null,
          diagnosis: diagnosis || null,
          medical_history: history || null,
          current_issue: currentIssue || null,
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      setRegistered(await res.json());
      setName("");
      setPatientId("");
      setAppointment("");
      setArrival(nowLocal());
      setDiagnosis("");
      setHistory("");
      setCurrentIssue("");
      setGolden(null);
      refreshQueue();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const triageOrder = [...queue].sort(
    (a, b) => a.priority - b.priority || a.arrival_datetime.localeCompare(b.arrival_datetime)
  );
  const newKey = registered ? rowKey(registered) : null;
  const newIndex = newKey ? triageOrder.findIndex((p) => rowKey(p) === newKey) : -1;

  return (
    <>
      <div className="panel">
        <div className="panel-head">New Patient Entry — Registration</div>
        <div className="panel-body">
          <GoldenSamples selected={golden} onPick={pickGolden} />
          <form className="form-grid" onSubmit={submit}>
            <div>
              <label>Patient Name (demo only, not persisted)</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Tan Ah Kow"
                maxLength={80}
              />
            </div>
            <div>
              <label>Patient ID (blank = auto-generate)</label>
              <input
                value={patientId}
                onChange={(e) => setPatientId(e.target.value.replace(/\D/g, ""))}
                placeholder="9-digit ID"
                inputMode="numeric"
              />
            </div>
            <div>
              <label>Clinic</label>
              <select value={clinic} onChange={(e) => { setClinic(e.target.value); setGolden(null); }}>
                {CLINICS.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Service Type</label>
              <select value={serviceType} onChange={(e) => { setServiceType(e.target.value); setGolden(null); }}>
                {SERVICE_TYPES.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Arrival (SGT)</label>
              <input
                type="datetime-local"
                value={arrival}
                onChange={(e) => { setArrival(e.target.value); setGolden(null); }}
                required
              />
            </div>
            <div>
              <label>Appointment (optional)</label>
              <input
                type="datetime-local"
                value={appointment}
                onChange={(e) => setAppointment(e.target.value)}
              />
            </div>
            <SuggestField
              id="diagnosis"
              label="Diagnosis"
              value={diagnosis}
              onChange={setDiagnosis}
              options={DIAGNOSIS_OPTIONS}
              placeholder="Pick a common diagnosis or type your own…"
              maxLength={500}
            />
            <SuggestField
              id="medical-history"
              label="Medical History"
              value={history}
              onChange={setHistory}
              options={HISTORY_OPTIONS}
              placeholder="Pick from the list or type your own…"
              maxLength={1000}
            />
            <SuggestField
              id="current-issue"
              label="Current Issue"
              value={currentIssue}
              onChange={setCurrentIssue}
              options={CURRENT_ISSUE_OPTIONS}
              placeholder="Pick a common presentation or type your own…"
              maxLength={1000}
            />
            <button className="primary" disabled={loading}>
              {loading ? "Registering patient & computing estimates…" : "Register Patient & Estimate Wait"}
            </button>
          </form>
          {error && <p style={{ color: "var(--mediq-red)", marginTop: 10 }}>{error}</p>}
        </div>
      </div>

      {registered && (
        <div className="panel">
          <div className="panel-head">Registration Ticket</div>
          <div className="panel-body" aria-live="polite">
            <div className={`estimate-card ${registered.status_band}`}>
              <div style={{ display: "flex", gap: 32, alignItems: "baseline" }}>
                <div>
                  <div className="muted">Queue position</div>
                  <div className="estimate-value">Q{registered.queue_position}</div>
                </div>
                <div>
                  <div className="muted">
                    {registered.model_estimate_min != null ? "Predicted wait — LightGBM model" : "Estimated wait"}
                  </div>
                  <div className={`estimate-value band-${registered.status_band}`}>
                    {Math.round(registered.estimated_wait_min)}
                    <span style={{ fontSize: 18 }}> min</span>
                    <span className={`band-chip ${registered.status_band}`}>{BAND_LABEL[registered.status_band]}</span>
                  </div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 6 }}>
                {registered.display_name ? `${registered.display_name} · ` : ""}
                Patient #{registered.patient_id} · {registered.service_type} ·{" "}
                {registered.clinic} · likely range {registered.range_min[0]}–{registered.range_min[1]} min
              </div>
              {registered.model_estimate_min != null && (
                <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <span
                    style={{
                      background: "var(--mediq-blue)", color: "#fff", borderRadius: 4,
                      padding: "2px 8px", fontSize: 12, fontWeight: 700, whiteSpace: "nowrap",
                    }}
                  >
                    ML model · LightGBM
                  </span>
                  <span style={{ fontSize: 13 }}>
                    typical error ±{registered.model_mae_min?.toFixed(1)} min
                    {modelCard && ` · trained on ${modelCard.train.events.toLocaleString()} historical events`}
                  </span>
                </div>
              )}
              {registered.model_breach_flag && (
                <div style={{ marginTop: 6, fontSize: 13, color: "var(--mediq-red)", fontWeight: 600 }}>
                  ⚠ May wait past the {registered.model_breach_limit_min}-minute limit for this service
                </div>
              )}
              {registered.rag_estimate_min != null && registered.rag_range_min && (
                <div className="muted" style={{ marginTop: 6, fontSize: 13 }}>
                  For comparison — {registered.rag_evidence_n} most similar historical cases: median{" "}
                  {Math.round(registered.rag_estimate_min)} min (range {Math.round(registered.rag_range_min[0])}–
                  {Math.round(registered.rag_range_min[1])} min)
                </div>
              )}
              <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <PriorityBadge p={registered.priority} label={registered.priority_label} />
                <span className="muted" style={{ fontSize: 13 }}>
                  {registered.triage_reasons.join(" · ")}
                </span>
              </div>
              {newIndex >= 0 && (
                <div style={{ marginTop: 8, fontSize: 13 }}>
                  <strong>{ordinal(newIndex + 1)} of {triageOrder.length}</strong> in today&apos;s triage order ·{" "}
                  <a
                    href={`#${NEW_ROW_ID}`}
                    style={{ color: "var(--mediq-blue)", fontWeight: 600 }}
                    onClick={(e) => {
                      e.preventDefault();
                      document.getElementById(NEW_ROW_ID)?.scrollIntoView({ behavior: "smooth", block: "center" });
                    }}
                  >
                    Show in list ↓
                  </a>
                </div>
              )}
              <p style={{ marginTop: 8 }}>{registered.explanation}</p>
            </div>
          </div>
        </div>
      )}

      {queue.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            Registered Today ({queue.length}) — triage order
          </div>
          <div className="panel-body">
            <table className="evidence-table">
              <thead>
                <tr>
                  <th>Priority</th>
                  <th>Q#</th>
                  <th>Patient</th>
                  <th>Clinic</th>
                  <th>Service</th>
                  <th>Pathway</th>
                  <th>Arrival</th>
                  <th>Est. wait</th>
                </tr>
              </thead>
              <tbody>
                {triageOrder.map((p, i) => {
                  const isNew = i === newIndex;
                  return (
                    <tr
                      key={rowKey(p)}
                      id={isNew ? NEW_ROW_ID : undefined}
                      aria-current={isNew ? "true" : undefined}
                      style={
                        isNew
                          ? { background: "var(--mediq-blue-l)", boxShadow: "inset 4px 0 0 var(--mediq-blue)" }
                          : undefined
                      }
                    >
                      <td>
                        <PriorityBadge p={p.priority} label={`P${p.priority}`} />
                      </td>
                      <td>Q{p.queue_position}</td>
                      <td style={isNew ? { fontWeight: 700 } : undefined}>
                        {p.display_name || `#${p.patient_id}`}
                        {isNew && (
                          <span
                            style={{
                              marginLeft: 8, background: "var(--mediq-blue)", color: "#fff", borderRadius: 4,
                              padding: "1px 7px", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.3px",
                            }}
                          >
                            NEW
                          </span>
                        )}
                      </td>
                      <td>{p.clinic}</td>
                      <td>{p.service_type}</td>
                      <td className="muted" style={{ fontSize: 13 }}>
                        {p.pathway.map((s) => s.station).join(" → ") || "—"}
                      </td>
                      <td>{p.arrival_datetime.slice(11, 16)}</td>
                      <td className={`band-${p.status_band}`} style={{ fontWeight: 600 }}>
                        {Math.round(p.estimated_wait_min)} min
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {modelCard && (
        <div className="panel">
          <button
            type="button"
            className="panel-head"
            aria-expanded={modelOpen}
            aria-controls="model-card-body"
            onClick={() => setModelOpen((o) => !o)}
            style={{
              width: "100%", border: "none", cursor: "pointer", textAlign: "left",
              display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12,
            }}
          >
            <span>
              Prediction Model — {modelCard.name} · typical error ±{modelCard.evaluation.app.MAE.toFixed(1)} min
            </span>
            <span aria-hidden="true">{modelOpen ? "▾ Hide" : "▸ Show"}</span>
          </button>
          {modelOpen && (
            <div className="panel-body" id="model-card-body">
              <ModelStats card={modelCard} />
            </div>
          )}
        </div>
      )}

      {registered && (
        <div className="panel">
          <div className="panel-head">Queue State at Arrival — HMM · LSTM</div>
          <div className="panel-body" aria-live="polite">
            {registered.model_unavailable_reason && (
              <p className="muted">{registered.model_unavailable_reason}</p>
            )}
            {registered.model_estimate_min != null && (
              <>
                {registered.model_is_analog_replay && (
                  <p className="qa-banner" style={{ marginBottom: 12 }}>
                    {registered.model_replay_note}
                  </p>
                )}
                <div style={{ display: "flex", gap: 40, marginBottom: 12, flexWrap: "wrap" }}>
                  <div>
                    <div className="muted">Queue depth now</div>
                    <div className="kpi-secondary">{registered.model_queue_depth} waiting</div>
                  </div>
                  <div>
                    <div className="muted">Congestion state (HMM)</div>
                    <div
                      className="kpi-secondary"
                      style={{ color: CONGESTION_COLOR[registered.model_congestion_state ?? "unknown"] }}
                    >
                      {registered.model_congestion_state}
                    </div>
                    <div className="muted" style={{ fontSize: 11 }}>
                      {registered.model_congestion_mean_wait_min != null
                        ? `state mean ${registered.model_congestion_mean_wait_min}m`
                        : "insufficient recent data"}
                    </div>
                  </div>
                  <div>
                    <div className="muted">Forecast trend (LSTM)</div>
                    <div className="kpi-secondary">
                      {TREND_ARROW[registered.model_forecast_trend ?? "no-data"]}
                    </div>
                  </div>
                </div>

                {registered.model_forecast.length > 0 && (
                  <>
                    <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                      Queue depth forecast — next{" "}
                      {registered.model_forecast[registered.model_forecast.length - 1].minutes_ahead} min
                    </div>
                    <div className="hourly-bars" style={{ maxWidth: 320 }}>
                      {registered.model_forecast.map((f, i) => {
                        const max = Math.max(...registered.model_forecast.map((x) => x.queue_depth), 1);
                        return (
                          <div
                            key={i}
                            className="hourly-bar"
                            role="img"
                            aria-label={`+${f.minutes_ahead} minutes: ${f.queue_depth} patients in queue`}
                            title={`+${f.minutes_ahead} min: ${f.queue_depth}`}
                            style={{
                              height: `${Math.max(4, (f.queue_depth / max) * 40)}px`,
                              background: "var(--mediq-blue)",
                            }}
                          />
                        );
                      })}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {registered && registered.pathway.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            Predicted Visit Pathway — {registered.pathway_label}
            {registered.total_visit_min != null &&
              ` · ~${Math.round(registered.total_visit_min)} min total`}
          </div>
          <div className="panel-body">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "stretch" }}>
              {registered.pathway.map((step, i) => (
                <div
                  key={i}
                  style={{
                    border: "1px solid var(--border, #ddd)",
                    borderRadius: 6,
                    padding: "8px 12px",
                    minWidth: 150,
                  }}
                >
                  <div style={{ fontWeight: 700 }}>
                    {i + 1}. {step.station}
                  </div>
                  <div className="muted" style={{ fontSize: 13 }}>
                    starts ~T+{Math.round(step.eta_offset_min)} min
                    <br />
                    wait ~{Math.round(step.expected_wait_min)} min · service ~
                    {Math.round(step.expected_service_min)} min
                    {step.note && (
                      <>
                        <br />
                        {step.note}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
              Pathway and durations from observed patient journeys (median per station).
            </p>
          </div>
        </div>
      )}

    </>
  );
}
