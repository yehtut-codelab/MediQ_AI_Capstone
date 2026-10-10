"use client";

import { useEffect, useState } from "react";
import SuggestField from "@/components/SuggestField";
import { BAND_LABEL } from "@/lib/band";
import { CURRENT_ISSUE_OPTIONS, DIAGNOSIS_OPTIONS, HISTORY_OPTIONS } from "@/lib/clinicalOptions";
import { nowLocal, upcomingAt } from "@/lib/datetime";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type DatasetRange = { min: string; max: string; categories: string[] };

type QueueSimulation = {
  clinic: string;
  category: string;
  as_of: string;
  replayed_at: string;
  is_analog_replay: boolean;
  replay_note: string | null;
  queue_depth: number;
  model_estimate_min: number;
  model_mae_min: number;
  status_band: "green" | "amber" | "red";
  congestion_state: "Low" | "Medium" | "High" | "unknown";
  congestion_mean_wait_min: number | null;
  congestion_observations: number;
  forecast: { minutes_ahead: number; queue_depth: number }[];
  forecast_trend: "rising" | "falling" | "stable" | "no-data";
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

type Estimate = {
  estimated_wait_min: number;
  range_min: [number, number];
  confidence: string;
  evidence_count: number;
  filters_widened: number;
  explanation: string;
  status_band: "green" | "amber" | "red";
  similar_events_sample: {
    score: number;
    rerank_score: number | null;
    service_point: string;
    wait_min: number;
    wait_start_iso: string;
  }[];
};

type SamplePatient = {
  patient_id: number;
  clinic: string;
  service_type: string;
  service_point: string;
  arrival_iso: string;
  actual_wait_min: number;
};

// The RAG evidence corpus (Qdrant) and the trained models (Models/) were built
// from two different exports with two different label schemes. This maps the
// model's category onto its RAG equivalent where one exists — categories with
// no entry here (refraction, laser_procedure, other) simply have no evidence
// corpus to search, so the estimate panel says so instead of erroring.
const CATEGORY_TO_SERVICE_TYPE: Record<string, string> = {
  consultation: "Consultation",
  visual_acuity: "VA",
  diagnostic_scan: "OCT",
  visual_field_test: "HVF",
  biometry: "Biometry",
  treatment: "Treatment",
};

const SERVICE_TYPE_TO_CATEGORY: Record<string, string> = {
  ...Object.fromEntries(Object.entries(CATEGORY_TO_SERVICE_TYPE).map(([cat, st]) => [st, cat])),
  Orthoptic: "other",
  "Financial Con": "other",
};

export default function ArrivalSimulator() {
  const [clinic, setClinic] = useState("TTSH Eye Centre");
  const [category, setCategory] = useState("consultation");
  const [arrival, setArrival] = useState(nowLocal);
  const [diagnosis, setDiagnosis] = useState("");
  const [history, setHistory] = useState("");
  const [currentIssue, setCurrentIssue] = useState("");
  const [sample, setSample] = useState<SamplePatient | null>(null);
  const [sampling, setSampling] = useState(false);
  const [loading, setLoading] = useState(false);

  const [range, setRange] = useState<DatasetRange | null>(null);

  const [result, setResult] = useState<Estimate | null>(null);
  const [error, setError] = useState("");
  const [ragUnavailable, setRagUnavailable] = useState<string | null>(null);

  const [simResult, setSimResult] = useState<QueueSimulation | null>(null);
  const [simError, setSimError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/v1/agent/dataset-range`)
      .then((r) => r.json())
      .then(setRange)
      .catch(() => {});
  }, []);

  async function loadSample() {
    setSampling(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/v1/arrivals/samples?n=1`);
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      const [s]: SamplePatient[] = await res.json();
      setClinic(s.clinic);
      setCategory(SERVICE_TYPE_TO_CATEGORY[s.service_type] ?? "other");
      // Samples are historical; keep the time of day but never preload a past date.
      setArrival(upcomingAt(s.arrival_iso));
      setSample(s);
      setResult(null);
      setSimResult(null);
      setRagUnavailable(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setSampling(false);
    }
  }

  async function runRag(serviceType: string | null, arrivalIso: string) {
    if (!serviceType) {
      setResult(null);
      setRagUnavailable(
        `"${category.replace(/_/g, " ")}" has no equivalent label in the RAG evidence corpus ` +
          `(Qdrant only covers Consultation, OCT, VA, HVF, Orthoptic, Financial Con) — no ` +
          `evidence-based estimate available for this category.`
      );
      return;
    }
    try {
      const res = await fetch(`${API}/api/v1/arrivals/estimate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clinic,
          service_type: serviceType,
          arrival_datetime: arrivalIso,
          diagnosis: diagnosis || null,
          medical_history: history || null,
          current_issue: currentIssue || null,
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      setResult(await res.json());
    } catch (err) {
      setResult(null);
      setError(String(err));
    }
  }

  async function runSim() {
    try {
      const res = await fetch(`${API}/api/v1/queue/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clinic, category, as_of: `${arrival}:00` }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      setSimResult(await res.json());
    } catch (err) {
      setSimResult(null);
      setSimError(String(err));
    }
  }

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSimError("");
    setRagUnavailable(null);

    const serviceType = CATEGORY_TO_SERVICE_TYPE[category] ?? null;
    const arrivalIso = `${arrival}:00+08:00`;

    await Promise.all([runRag(serviceType, arrivalIso), runSim()]);
    setLoading(false);
  }

  return (
    <>
      <div className="panel">
        <div className="panel-head">New Patient Arrival</div>
        <div className="panel-body">
          <p className="muted" style={{ marginTop: 0, marginBottom: 12 }}>
            One patient scenario drives two independent estimates below: a RAG search over similar
            historical cases, and a queue simulation from the trained XGBoost / HMM / LSTM models.
            The RAG estimate is unavailable for a few categories the evidence corpus doesn't cover.
            The queue simulator can use any date — for one outside its trained dataset, it replays
            the nearest historical date with the same weekday and time instead.
          </p>
          <form className="form-grid" onSubmit={run}>
            <div>
              <label>Clinic</label>
              <select
                value={clinic}
                onChange={(e) => {
                  setClinic(e.target.value);
                  setSample(null);
                }}
              >
                <option>TTSH Eye Centre</option>
                <option>Clinic 1A</option>
                {sample && !["TTSH Eye Centre", "Clinic 1A"].includes(sample.clinic) && (
                  <option>{sample.clinic}</option>
                )}
              </select>
            </div>
            <div>
              <label>Service Category</label>
              <select
                value={category}
                onChange={(e) => {
                  setCategory(e.target.value);
                  setSample(null);
                }}
              >
                {(range?.categories ?? [category]).map((c) => (
                  <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
                ))}
              </select>
            </div>
            <div>
              <label>
                Arrival (SGT) {range && `— trained-model data covers ${range.min.slice(0, 10)} … ${range.max.slice(0, 10)}`}
              </label>
              <input
                type="datetime-local"
                value={arrival}
                onChange={(e) => {
                  setArrival(e.target.value);
                  setSample(null);
                }}
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
              {loading ? "Estimating & simulating…" : "Estimate & Simulate"}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={sampling}
              onClick={loadSample}
            >
              {sampling ? "Loading sample…" : "Load Sample Patient"}
            </button>
          </form>
          {sample && (
            <p className="muted" style={{ marginTop: 10 }}>
              Sample patient #{sample.patient_id} · historical station {sample.service_point} ·
              actually waited {sample.actual_wait_min.toFixed(0)} min · originally arrived{" "}
              {sample.arrival_iso.slice(0, 16).replace("T", " ")}, shown at the same time today or
              tomorrow
            </p>
          )}
          {error && <p style={{ color: "var(--mediq-red)", marginTop: 10 }}>{error}</p>}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Evidence-Based Estimate — RAG</div>
        <div className="panel-body" aria-live="polite">
          {ragUnavailable && <p className="muted">{ragUnavailable}</p>}
          {!ragUnavailable && !result && !loading && (
            <p className="muted">Run "Estimate &amp; Simulate" above to see a result.</p>
          )}
          {result && (
            <>
              <div className={`estimate-card ${result.status_band}`}>
                <div className={`estimate-value band-${result.status_band}`}>
                  {Math.round(result.estimated_wait_min)}
                  <span style={{ fontSize: 18 }}> min</span>
                  <span className={`band-chip ${result.status_band}`}>{BAND_LABEL[result.status_band]}</span>
                </div>
                <div className="muted">
                  Range {result.range_min[0]}–{result.range_min[1]} min · confidence{" "}
                  {result.confidence} · {result.evidence_count} similar events
                  {result.filters_widened > 0 && ` · filters widened ×${result.filters_widened}`}
                  {sample && (
                    <> · Actual historical wait: {sample.actual_wait_min.toFixed(0)} min</>
                  )}
                </div>
                <p style={{ marginTop: 8 }}>{result.explanation}</p>
              </div>

              <p className="muted" style={{ marginTop: 12, marginBottom: 4 }}>
                Top {result.similar_events_sample.length} similar historical cases
                (vector search → cross-encoder rerank)
              </p>
              <table className="evidence-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Rerank</th>
                    <th>Vector</th>
                    <th>When</th>
                    <th>Station</th>
                    <th>Waited</th>
                  </tr>
                </thead>
                <tbody>
                  {result.similar_events_sample.map((ev, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{ev.rerank_score != null ? ev.rerank_score.toFixed(3) : "—"}</td>
                      <td>{ev.score.toFixed(3)}</td>
                      <td>{ev.wait_start_iso.slice(0, 16).replace("T", " ")}</td>
                      <td>{ev.service_point}</td>
                      <td>{ev.wait_min.toFixed(0)} min</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">Simulated Queue State — XGBoost · HMM · LSTM</div>
        <div className="panel-body" aria-live="polite">
          {simError && <p style={{ color: "var(--mediq-red)" }}>{simError}</p>}
          {!simError && !simResult && !loading && (
            <p className="muted">Run "Estimate &amp; Simulate" above to see a result.</p>
          )}
          {simResult && (
            <>
              {simResult.is_analog_replay && (
                <p className="qa-banner" style={{ marginBottom: 12 }}>
                  {simResult.replay_note}
                </p>
              )}
              <div style={{ display: "flex", gap: 40, marginBottom: 12, flexWrap: "wrap" }}>
                <div>
                  <div className="muted">Model wait estimate</div>
                  <div className={`estimate-value band-${simResult.status_band}`}>
                    {Math.round(simResult.model_estimate_min)}
                    <span style={{ fontSize: 18 }}> min</span>
                    <span className={`band-chip ${simResult.status_band}`}>{BAND_LABEL[simResult.status_band]}</span>
                  </div>
                  <div className="muted" style={{ fontSize: 11 }}>
                    XGBoost · val MAE {simResult.model_mae_min} min
                  </div>
                </div>
                <div>
                  <div className="muted">Queue depth now</div>
                  <div className="kpi-secondary">{simResult.queue_depth} waiting</div>
                </div>
                <div>
                  <div className="muted">Congestion state (HMM)</div>
                  <div
                    className="kpi-secondary"
                    style={{ color: CONGESTION_COLOR[simResult.congestion_state] }}
                  >
                    {simResult.congestion_state}
                  </div>
                  <div className="muted" style={{ fontSize: 11 }}>
                    {simResult.congestion_mean_wait_min != null
                      ? `state mean ${simResult.congestion_mean_wait_min}m · ${simResult.congestion_observations} recent waits`
                      : "insufficient recent data"}
                  </div>
                </div>
                <div>
                  <div className="muted">Forecast trend (LSTM)</div>
                  <div className="kpi-secondary">{TREND_ARROW[simResult.forecast_trend]}</div>
                </div>
              </div>

              {simResult.forecast.length > 0 && (
                <>
                  <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                    Queue depth forecast — next {simResult.forecast[simResult.forecast.length - 1].minutes_ahead} min
                  </div>
                  <div className="hourly-bars" style={{ maxWidth: 320 }}>
                    {simResult.forecast.map((f, i) => {
                      const max = Math.max(...simResult.forecast.map((x) => x.queue_depth), 1);
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
                  <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                    {simResult.forecast.map((f) => `+${f.minutes_ahead}m: ${f.queue_depth}`).join(" · ")}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
