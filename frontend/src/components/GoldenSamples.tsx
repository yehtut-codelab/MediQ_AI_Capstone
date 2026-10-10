"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// A curated demo patient. Its clinic state is pinned to `replay_at`, a recorded
// moment in the historical dataset, so every model output is present and the
// result is the same on any day it is presented.
export type GoldenSample = {
  id: string;
  label: string;
  story: string;
  display_name: string;
  clinic: string;
  service_type: string;
  category: string;
  arrival_time: string; // "HH:MM"
  appointment_time: string; // "HH:MM"
  replay_at: string;
  diagnosis: string;
  medical_history: string;
  current_issue: string;
};

const minutes = (hhmm: string) => {
  const [h, m] = hhmm.split(":").map(Number);
  return h * 60 + m;
};

/** Arrival minus appointment, in minutes (negative = arrived early). */
export const appointmentLagMin = (s: GoldenSample) => minutes(s.arrival_time) - minutes(s.appointment_time);

export function GoldenSamples({
  selected,
  onPick,
}: {
  selected: GoldenSample | null;
  onPick: (sample: GoldenSample) => void;
}) {
  const [samples, setSamples] = useState<GoldenSample[]>([]);

  useEffect(() => {
    fetch(`${API}/api/v1/samples/golden`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setSamples)
      .catch(() => {});
  }, []);

  if (samples.length === 0) return null;
  return (
    <div className="golden" role="group" aria-label="Golden demo samples">
      <div className="golden-head">Golden samples — pick one to fill the form with today&apos;s date</div>
      <div className="golden-grid">
        {samples.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className={selected?.id === s.id ? "golden-card active" : "golden-card"}
            aria-pressed={selected?.id === s.id}
            onClick={() => onPick(s)}
          >
            <span className="golden-label">{i + 1}. {s.label}</span>
            <span className="golden-story">{s.story}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
