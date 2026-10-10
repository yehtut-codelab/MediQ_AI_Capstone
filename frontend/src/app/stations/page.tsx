"use client";

import { useEffect, useState } from "react";
import { BAND_LABEL } from "@/lib/band";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Hourly = { hour: number; median_wait_min: number; count: number };

type Station = {
  service_type: string;
  count: number;
  median_wait_min: number;
  p75_wait_min: number;
  p90_wait_min: number;
  band: "green" | "amber" | "red";
  registered_today: number;
  hourly: Hourly[];
};

type Overview = Record<string, Station[]>;

// Tab value for the combined view; not a clinic name.
const ALL = "__all__";

const BAND_COLOR = {
  green: "var(--mediq-green)",
  amber: "var(--mediq-gold)",
  red: "var(--mediq-red)",
} as const;

function HourlyBars({ hourly }: { hourly: Hourly[] }) {
  if (hourly.length === 0) return null;
  const max = Math.max(...hourly.map((h) => h.median_wait_min));
  return (
    <div>
      <div className="hourly-bars">
        {hourly.map((h) => (
          <div
            key={h.hour}
            className="hourly-bar"
            role="img"
            aria-label={`${String(h.hour).padStart(2, "0")}:00 — median ${h.median_wait_min} minutes (${h.count} events)`}
            title={`${String(h.hour).padStart(2, "0")}:00 — median ${h.median_wait_min} min (${h.count} events)`}
            style={{
              height: `${Math.max(4, (h.median_wait_min / max) * 40)}px`,
              background:
                h.median_wait_min >= 60
                  ? "var(--mediq-red)"
                  : h.median_wait_min >= 30
                    ? "var(--mediq-gold)"
                    : "var(--mediq-green)",
            }}
          />
        ))}
      </div>
      <div className="hourly-axis">
        <span>{String(hourly[0].hour).padStart(2, "0")}:00</span>
        <span>{String(hourly[hourly.length - 1].hour).padStart(2, "0")}:00</span>
      </div>
    </div>
  );
}

export default function StationsDashboard() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [clinic, setClinic] = useState(ALL);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/v1/stations/overview`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data: Overview = await res.json();
        setOverview(data);
      })
      .catch((err) => setError(String(err)));
  }, []);

  if (error) return <p style={{ color: "var(--mediq-red)" }}>{error}</p>;
  if (!overview)
    return <p className="muted">Computing station statistics from historical wait events… (first load takes a few seconds)</p>;

  const stationList =
    clinic === ALL
      ? Object.entries(overview).flatMap(([c, list]) => list.map((s) => ({ ...s, clinic: c })))
      : (overview[clinic] ?? []).map((s) => ({ ...s, clinic }));
  const totalStations = Object.values(overview).reduce((n, list) => n + list.length, 0);
  const totalEvents = Object.values(overview).flat().reduce((n, st) => n + st.count, 0);

  return (
    <>
      <div className="panel">
        <div className="panel-head">Station Overview — Historical Wait Profile by Service</div>
        <div className="panel-body">
          <div className="clinic-tabs">
            <button className={clinic === ALL ? "tab active" : "tab"} onClick={() => setClinic(ALL)}>
              All stations ({totalStations})
            </button>
            {Object.keys(overview).map((c) => (
              <button
                key={c}
                className={c === clinic ? "tab active" : "tab"}
                onClick={() => setClinic(c)}
              >
                {c} ({overview[c].length} stations)
              </button>
            ))}
          </div>

          <div className="station-grid" aria-live="polite">
            {stationList.map((s) => (
              <div
                key={`${s.clinic}-${s.service_type}`}
                className="station-card"
                style={{ borderTopColor: BAND_COLOR[s.band] }}
              >
                <div className="station-name">
                  {s.service_type}
                  <span className={`band-chip ${s.band}`}>{BAND_LABEL[s.band]}</span>
                </div>
                {clinic === ALL && (
                  <div className="muted" style={{ fontSize: 11 }}>{s.clinic}</div>
                )}
                <div className="station-median" style={{ color: BAND_COLOR[s.band] }}>
                  {s.median_wait_min}
                  <span style={{ fontSize: 13, fontWeight: 400 }}> min median</span>
                </div>
                <div className="muted" style={{ fontSize: 11 }}>
                  P75 {s.p75_wait_min}m · P90 {s.p90_wait_min}m · {s.count.toLocaleString()} events
                </div>
                {s.registered_today > 0 && (
                  <div className="station-reg">
                    {s.registered_today} registered today
                  </div>
                )}
                <HourlyBars hourly={s.hourly} />
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="muted" style={{ marginTop: -6 }}>
        Based on {totalEvents.toLocaleString()} historical wait events across {totalStations} stations;
        stations with fewer than 200 events are not shown. Bars show median wait by hour of day
        (07:00–18:00). Colors follow clinic thresholds: green &lt;30 min, amber 30–60 min, red &gt;60 min.
      </p>
    </>
  );
}
