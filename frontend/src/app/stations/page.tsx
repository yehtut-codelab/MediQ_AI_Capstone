"use client";

import { useEffect, useState } from "react";
import { BAND_LABEL, bandFor, type Band } from "@/lib/band";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Hourly = {
  hour: number;
  median_wait_min: number;
  p75_wait_min: number;
  p90_wait_min: number;
  count: number;
};

type Profile = {
  count: number;
  days: number;
  median_wait_min: number;
  p75_wait_min: number;
  p90_wait_min: number;
  band: Band;
  hourly: Hourly[];
};

type Station = {
  category: string;
  count: number;
  registered_today: number;
  // "all" (every day pooled) plus "0".."6" (Monday = 0) where there is enough data.
  profiles: Record<string, Profile>;
};

// Hold-out accuracy of this page's method.
type Accuracy = {
  mae_min: number;
  rmse_min: number;
  r2: number;
  train_cutoff: string;
  train_events: number;
  validation_events: number;
};

type Weekly = {
  date_from: string;
  date_to: string;
  total_events: number;
  weekdays: number[];
  accuracy: Accuracy;
  clinics: Record<string, Station[]>;
};

// Tab values for the combined views; not a clinic name or a weekday.
const ALL = "__all__";
const ALL_DAYS = "all";

const DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const HOURS = Array.from({ length: 12 }, (_, i) => i + 7); // 07:00–18:00

const BAND_COLOR = {
  green: "var(--mediq-green)",
  amber: "var(--mediq-gold)",
  red: "var(--mediq-red)",
} as const;

const hh = (hour: number) => `${String(hour).padStart(2, "0")}:00`;

// Monday = 0, to match the backend's weekday numbering.
const weekdayOf = (d: Date) => (d.getDay() + 6) % 7;

/** The next calendar date that falls on `weekday`, counting today. */
function nextDate(weekday: number): string {
  const d = new Date();
  d.setDate(d.getDate() + ((weekday - weekdayOf(d) + 7) % 7));
  return d.toLocaleDateString("en-SG", { day: "numeric", month: "short" });
}

function HourlyBars({ hourly }: { hourly: Hourly[] }) {
  if (hourly.length === 0) return null;
  const byHour = new Map(hourly.map((h) => [h.hour, h]));
  const max = Math.max(...hourly.map((h) => h.p90_wait_min), 1);
  return (
    <div>
      <div className="hourly-range-bars">
        {HOURS.map((hour) => {
          const h = byHour.get(hour);
          if (!h) return <div key={hour} className="hourly-col" />;
          const label = `${hh(hour)} — median ${h.median_wait_min} min, 9 in 10 within ${h.p90_wait_min} min (${h.count.toLocaleString()} events)`;
          const color = BAND_COLOR[bandFor(h.median_wait_min)];
          return (
            <div key={hour} className="hourly-col" role="img" aria-label={label} title={label}>
              <div
                className="hourly-range"
                style={{ height: `${Math.max(4, (h.p90_wait_min / max) * 100)}%`, background: color }}
              />
              <div
                className="hourly-median"
                style={{ height: `${Math.max(4, (h.median_wait_min / max) * 100)}%`, background: color }}
              />
            </div>
          );
        })}
      </div>
      <div className="hourly-axis">
        <span>{hh(HOURS[0])}</span>
        <span>{hh(HOURS[HOURS.length - 1])}</span>
      </div>
    </div>
  );
}

const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-SG", { day: "numeric", month: "short", year: "numeric" });

function MethodPanel({ weekly }: { weekly: Weekly }) {
  const a = weekly.accuracy;
  return (
    <div className="panel">
      <div className="panel-head">How These Figures Are Calculated — Method and Accuracy</div>
      <div className="panel-body">
        <div className="method-callout">
          <div className="method-label">Method used on this page</div>
          <div className="method-name">Historical median by station, day of week and hour</div>
          <div>
            A statistical pattern from past visits, <strong>not a machine-learning forecast</strong>. It
            shows what the coming week usually looks like; it does not react to today&apos;s queue.
          </div>
        </div>

        <div className="method-stats">
          <div>
            <div className="muted">Historical wait events</div>
            <div className="kpi-secondary">{weekly.total_events.toLocaleString()}</div>
            <div className="muted">{fmtDate(weekly.date_from)} – {fmtDate(weekly.date_to)}</div>
          </div>
          <div>
            <div className="muted">Typical error (MAE)</div>
            <div className="kpi-secondary" style={{ color: "var(--mediq-blue)" }}>±{a.mae_min.toFixed(1)} min</div>
            <div className="muted">per patient wait</div>
          </div>
          <div>
            <div className="muted">RMSE</div>
            <div className="kpi-secondary">{a.rmse_min.toFixed(1)} min</div>
          </div>
          <div>
            <div className="muted">R² (variation explained)</div>
            <div className="kpi-secondary">{a.r2.toFixed(2)}</div>
          </div>
        </div>

        <p className="method-note">
          <strong>How accuracy was measured:</strong> the medians were fitted on the{" "}
          {a.train_events.toLocaleString()} events up to {fmtDate(a.train_cutoff)} and tested on the{" "}
          {a.validation_events.toLocaleString()} events after that date. The error is how far an
          individual patient&apos;s wait was from the median for their station, weekday and arrival hour.
        </p>

        <ul className="method-list">
          <li><strong>Solid bars:</strong> median wait for each hour of the day (07:00–18:00).</li>
          <li><strong>Pale bars:</strong> 90th percentile for that hour, the wait on a bad day.</li>
          <li><strong>Row under each chart:</strong> median wait in minutes for each day of the week.</li>
          <li><strong>Not shown:</strong> hours with fewer than 20 events and days with fewer than 50.</li>
          <li>
            <strong>Colors:</strong> <span style={{ color: BAND_COLOR.green }}>green</span> under 30 min,{" "}
            <span style={{ color: BAND_COLOR.amber }}>amber</span> 30–60 min,{" "}
            <span style={{ color: BAND_COLOR.red }}>red</span> over 60 min.
          </li>
        </ul>
      </div>
    </div>
  );
}

export default function StationsDashboard() {
  const [weekly, setWeekly] = useState<Weekly | null>(null);
  const [clinic, setClinic] = useState(ALL);
  // null until the user picks a day; the default depends on the loaded data.
  const [pickedDay, setPickedDay] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/v1/stations/weekly`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data: Weekly = await res.json();
        setWeekly(data);
      })
      .catch((err) => setError(String(err)));
  }, []);

  if (error) return <p style={{ color: "var(--mediq-red)" }}>{error}</p>;
  if (!weekly)
    return <p className="muted">Loading historical wait patterns… (first load takes a few seconds)</p>;

  const allStations = Object.values(weekly.clinics).flat();
  const today = weekdayOf(new Date());
  // Days in calendar order starting from today, so the tabs read as the week ahead.
  const upcoming = [...weekly.weekdays].sort((a, b) => ((a - today + 7) % 7) - ((b - today + 7) % 7));
  // Default to the first upcoming day that most stations have a pattern for
  // (skips e.g. a Saturday on which only a few stations run).
  const defaultDay =
    upcoming.find((d) => allStations.filter((s) => s.profiles[String(d)]).length * 2 >= allStations.length) ??
    upcoming[0];
  const day = pickedDay ?? String(defaultDay);

  const stationList = (
    clinic === ALL
      ? Object.entries(weekly.clinics).flatMap(([c, list]) => list.map((s) => ({ ...s, clinic: c })))
      : (weekly.clinics[clinic] ?? []).map((s) => ({ ...s, clinic }))
  ).sort(
    // Slowest first on the selected day; stations with no data that day go last.
    (a, b) => (b.profiles[day]?.median_wait_min ?? -1) - (a.profiles[day]?.median_wait_min ?? -1),
  );
  const totalStations = allStations.length;
  const dayName = day === ALL_DAYS ? "" : DAY_LONG[Number(day)];

  return (
    <>
      <div className="panel">
        <div className="panel-head">Station Overview — Expected Wait Pattern by Day of Week</div>
        <div className="panel-body">
          <div className="clinic-tabs">
            <button className={clinic === ALL ? "tab active" : "tab"} onClick={() => setClinic(ALL)}>
              All stations ({totalStations})
            </button>
            {Object.keys(weekly.clinics).map((c) => (
              <button
                key={c}
                className={c === clinic ? "tab active" : "tab"}
                onClick={() => setClinic(c)}
              >
                {c} ({weekly.clinics[c].length} stations)
              </button>
            ))}
          </div>

          <div className="clinic-tabs" role="group" aria-label="Day of week">
            {upcoming.map((d) => (
              <button
                key={d}
                className={String(d) === day ? "tab active" : "tab"}
                aria-pressed={String(d) === day}
                onClick={() => setPickedDay(String(d))}
              >
                {DAY_SHORT[d]} {nextDate(d)}
                {d === today && " · today"}
              </button>
            ))}
            <button
              className={day === ALL_DAYS ? "tab active" : "tab"}
              aria-pressed={day === ALL_DAYS}
              onClick={() => setPickedDay(ALL_DAYS)}
            >
              All days
            </button>
          </div>

          <div className="station-grid" aria-live="polite">
            {stationList.map((s) => {
              const p = s.profiles[day];
              return (
                <div
                  key={`${s.clinic}-${s.category}`}
                  className="station-card"
                  style={p && { borderTopColor: BAND_COLOR[p.band] }}
                >
                  <div className="station-name">
                    {s.category.replace(/_/g, " ")}
                    {p && <span className={`band-chip ${p.band}`}>{BAND_LABEL[p.band]}</span>}
                  </div>
                  {clinic === ALL && (
                    <div className="muted" style={{ fontSize: 11 }}>{s.clinic}</div>
                  )}
                  {p ? (
                    <>
                      <div className="station-median" style={{ color: BAND_COLOR[p.band] }}>
                        {p.median_wait_min}
                        <span style={{ fontSize: 13, fontWeight: 400 }}> min median</span>
                      </div>
                      <div className="muted" style={{ fontSize: 11 }}>
                        P75 {p.p75_wait_min}m · P90 {p.p90_wait_min}m · {p.count.toLocaleString()} events
                        {day !== ALL_DAYS && ` over ${p.days} ${dayName}s`}
                      </div>
                    </>
                  ) : (
                    <div className="muted" style={{ fontSize: 12, margin: "8px 0" }}>
                      Too few visits on {dayName}s to show a pattern.
                    </div>
                  )}
                  {s.registered_today > 0 && (
                    <div className="station-reg">
                      {s.registered_today} registered today
                    </div>
                  )}
                  {p && <HourlyBars hourly={p.hourly} />}
                  <div className="week-strip" role="group" aria-label="Median wait by day of week">
                    {weekly.weekdays.map((d) => {
                      const dp = s.profiles[String(d)];
                      return (
                        <button
                          key={d}
                          className={String(d) === day ? "week-cell active" : "week-cell"}
                          onClick={() => setPickedDay(String(d))}
                          title={
                            dp
                              ? `${DAY_LONG[d]} — median ${dp.median_wait_min} min, P90 ${dp.p90_wait_min} min`
                              : `${DAY_LONG[d]} — too few visits`
                          }
                        >
                          <span>{DAY_SHORT[d]}</span>
                          <b style={dp && { color: BAND_COLOR[dp.band] }}>
                            {dp ? Math.round(dp.median_wait_min) : "–"}
                          </b>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <MethodPanel weekly={weekly} />
    </>
  );
}
