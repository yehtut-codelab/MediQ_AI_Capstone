"""Back-test: can the deployed LSTM queue forecast warn of congestion in advance?

Replays the hold-out period (after the models' train cut-off) through the same
feature pipeline the app uses (app.services.livestate.lstm_bucket_features,
re-implemented here in vectorised form) and measures:

  1. Forecast error per horizon (15/30/45/60 min) against two baselines:
     persistence (queue stays as it is now) and the historical profile
     (training-period mean depth for that station, weekday and time of day —
     the kind of information the Station Overview page shows today).
  2. Early warning of congestion onsets. A station is "congested" when its
     queue depth reaches its own 90th-percentile level from the training
     period. At each 15-minute decision point where the station is NOT yet
     congested, an alert is raised if the forecast for any of the next four
     buckets reaches that level. Reported: precision, recall, lead time, and
     the same for baselines given the same number of alerts.

Two LSTM variants are run because the app's replay computes "mean waiting time
of patients who arrived in the bucket" from their FINAL waiting times, which a
live system would not yet know for patients still waiting:

  * as-deployed : exactly what the app does on replay (uses future information)
  * leak-free   : still-waiting patients contribute their elapsed wait so far

Usage (from backend/):
    .venv/Scripts/python.exe scripts/backtest_early_warning.py [--out results.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import livestate, models_service  # noqa: E402

MIN = np.timedelta64(1, "m").astype("timedelta64[ns]").astype(np.int64)
BUCKET = 15 * MIN
HALF = BUCKET // 2
DAY_START_MIN = 8 * 60        # first decision point 08:00
N_SLOTS = 37                  # 08:00 … 17:00 every 15 min
PCTL = 90                     # congestion level = this percentile of training depth
MIN_LEVEL = 5                 # …but never below this many patients
MAIN_STATION_EVENTS = 15_000  # stations pooled in the headline numbers


def ns(ts) -> np.ndarray:
    return np.asarray(ts, dtype="datetime64[ns]").astype(np.int64)


class Station:
    def __init__(self, df: pd.DataFrame):
        d = df.sort_values("waiting_start")
        self.ws = ns(d["waiting_start"].values)
        self.ss_by_ws = ns(d["service_start"].values)
        self.ss = np.sort(self.ss_by_ws)
        self.wait = d["waiting_time_min"].to_numpy(dtype=float)
        self.cum_wait = np.concatenate([[0.0], np.cumsum(self.wait)])

    def depth(self, t: np.ndarray) -> np.ndarray:
        """Patients whose wait has started but service has not, at each t."""
        return np.searchsorted(self.ws, t, "right") - np.searchsorted(self.ss, t, "right")


def decision_times(days: np.ndarray) -> np.ndarray:
    """(n_days, N_SLOTS) decision timestamps in ns."""
    base = ns(days)[:, None] + DAY_START_MIN * MIN
    return base + np.arange(N_SLOTS)[None, :] * BUCKET


def build_features(st: Station, t: np.ndarray, cat_enc: int, steps: int, leak_free: bool) -> np.ndarray:
    """(len(t), steps, 8) raw LSTM features, same definitions as livestate."""
    feats = np.zeros((len(t), steps, 8))
    tt = pd.to_datetime(t)
    for col, i in enumerate(range(steps, 0, -1)):
        b_end = t - BUCKET * (i - 1)
        b_start = b_end - BUCKET
        lo = np.searchsorted(st.ws, b_start, "left")
        hi = np.searchsorted(st.ws, b_end, "left")
        n_arr = hi - lo
        if leak_free:
            wmean = np.zeros(len(t))
            for r in np.nonzero(n_arr)[0]:
                sl = slice(lo[r], hi[r])
                elapsed = (t[r] - st.ws[sl]) / MIN
                seen = np.where(st.ss_by_ws[sl] <= t[r], st.wait[sl], elapsed)
                wmean[r] = seen.mean()
        else:
            wmean = np.divide(st.cum_wait[hi] - st.cum_wait[lo], n_arr,
                              out=np.zeros(len(t)), where=n_arr > 0)
        bs = tt - pd.to_timedelta((BUCKET * i) / MIN, unit="m")
        feats[:, col, 0] = st.depth(b_start + HALF)
        feats[:, col, 1] = n_arr
        feats[:, col, 2] = wmean
        feats[:, col, 3] = bs.hour
        feats[:, col, 4] = bs.dayofweek
        feats[:, col, 5] = bs.month
        feats[:, col, 6] = cat_enc
        feats[:, col, 7] = (bs.dayofweek >= 5).astype(int)
    return feats


def lstm_forecast(feats: np.ndarray) -> np.ndarray:
    model, scaler = models_service._lstm()
    n, steps, f = feats.shape
    scaled = scaler.transform(feats.reshape(-1, f)).reshape(n, steps, f)
    with torch.no_grad():
        out = model(torch.tensor(scaled, dtype=torch.float32)).numpy()
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    return np.clip(out * (hi - lo) + lo, 0, None)


def prf(alert: np.ndarray, label: np.ndarray) -> dict:
    tp = int((alert & label).sum())
    n_alert, n_pos = int(alert.sum()), int(label.sum())
    return {
        "alerts": n_alert,
        "precision": round(tp / n_alert, 3) if n_alert else None,
        "recall": round(tp / n_pos, 3) if n_pos else None,
    }


def top_n_alerts(score: np.ndarray, n: int) -> np.ndarray:
    """Alert on the n highest scores — gives a baseline the same alert budget."""
    alert = np.zeros(len(score), dtype=bool)
    if n > 0:
        alert[np.argsort(-score, kind="stable")[:n]] = True
    return alert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = models_service.lstm_meta()
    steps, horizon = meta["lookback_steps"], meta["horizon_steps"]
    cutoff = pd.Timestamp(meta["train_cutoff"]) + pd.Timedelta(days=1)
    cat_codes = models_service.wait_meta()["category_encoding"]

    ev = livestate._events()
    ev = ev.assign(day=ev["waiting_start"].dt.normalize())
    hold_lo, hold_hi = cutoff, ev["waiting_start"].max()
    print(f"Train period : {ev['waiting_start'].min():%Y-%m-%d} … {cutoff - pd.Timedelta(days=1):%Y-%m-%d}")
    print(f"Hold-out     : {hold_lo:%Y-%m-%d} … {hold_hi:%Y-%m-%d}")

    rows, pooled = [], {k: [] for k in
                        ["label", "eligible", "d0", "truth", "f_dep", "f_free", "hist", "trend", "main"]}
    episodes = []  # (station, main, lead_deployed, lead_leakfree) per congestion onset

    for (clinic, cat), sdf in ev.groupby(["clinic_name", "service_category"]):
        st = Station(sdf)
        counts = sdf.groupby("day").size()
        active = counts[counts >= 20].index
        train_days = active[active < cutoff].values
        test_days = active[active >= cutoff].values
        if len(train_days) < 30 or len(test_days) < 10:
            continue
        is_main = len(sdf) >= MAIN_STATION_EVENTS

        # historical profile + congestion level, from the training period only
        mids_train = decision_times(train_days) + HALF                 # (days, slots)
        mids_train = np.concatenate(
            [mids_train, mids_train[:, -1:] + BUCKET * np.arange(1, horizon + 1)[None, :]], axis=1)
        depth_train = st.depth(mids_train.ravel()).reshape(mids_train.shape)
        level = max(MIN_LEVEL, float(np.percentile(depth_train[:, :N_SLOTS], PCTL)))
        dow_train = pd.to_datetime(train_days).dayofweek.values
        profile = np.zeros((7, mids_train.shape[1]))
        for d in range(7):
            if (dow_train == d).any():
                profile[d] = depth_train[dow_train == d].mean(axis=0)

        # hold-out decision points
        T = decision_times(test_days)                                   # (days, slots)
        t = T.ravel()
        d0 = st.depth(t)
        truth = np.stack([st.depth(t + BUCKET * k - HALF) for k in range(1, horizon + 1)], axis=1)
        dow = np.repeat(pd.to_datetime(test_days).dayofweek.values, N_SLOTS)
        slot = np.tile(np.arange(N_SLOTS), len(test_days))
        hist = np.stack([profile[dow, slot + k - 1] for k in range(1, horizon + 1)], axis=1)
        trend = d0 + (d0 - st.depth(t - 2 * BUCKET))

        enc = cat_codes.get(cat, cat_codes["other"])
        feats_dep = build_features(st, t, enc, steps, leak_free=False)
        feats_free = build_features(st, t, enc, steps, leak_free=True)
        active_pt = (feats_dep[:, :, 0].sum(axis=1) + feats_dep[:, :, 1].sum(axis=1)) > 0
        f_dep, f_free = lstm_forecast(feats_dep), lstm_forecast(feats_free)

        eligible = active_pt & (d0 < level)
        label = truth.max(axis=1) >= level

        a_dep = eligible & (f_dep.max(axis=1) >= level)
        a_free = eligible & (f_free.max(axis=1) >= level)
        pos = eligible & label
        row = {
            "clinic": clinic, "category": cat, "events": len(sdf), "main": is_main,
            "congestion_level": round(level, 1),
            "decision_points": int(eligible.sum()), "onsets_within_1h": int(pos.sum()),
            "mae_lstm_deployed": [round(float(np.abs(f_dep[active_pt, k] - truth[active_pt, k]).mean()), 2) for k in range(horizon)],
            "mae_lstm_leakfree": [round(float(np.abs(f_free[active_pt, k] - truth[active_pt, k]).mean()), 2) for k in range(horizon)],
            "mae_persistence": [round(float(np.abs(d0[active_pt] - truth[active_pt, k]).mean()), 2) for k in range(horizon)],
            "mae_hist_profile": [round(float(np.abs(hist[active_pt, k] - truth[active_pt, k]).mean()), 2) for k in range(horizon)],
            "alert_deployed": prf(a_dep[eligible], label[eligible]),
            "alert_leakfree": prf(a_free[eligible], label[eligible]),
        }
        rows.append(row)

        # congestion onsets on the mid-bucket series, and how early each was flagged
        n_days = len(test_days)
        mid_depth = st.depth((T + HALF).ravel()).reshape(n_days, N_SLOTS)
        el2, ad2, af2 = (x.reshape(n_days, N_SLOTS) for x in (eligible, a_dep, a_free))
        for di in range(n_days):
            for j in range(1, N_SLOTS):
                if mid_depth[di, j] >= level and mid_depth[di, j - 1] < level:
                    leads = {}
                    for name, al in (("dep", ad2), ("free", af2)):
                        lead = None
                        for s in range(max(0, j - horizon + 1), j + 1):   # earliest first
                            if el2[di, s] and al[di, s]:
                                lead = 7.5 + 15 * (j - s)
                                break
                        leads[name] = lead
                    episodes.append((f"{clinic}/{cat}", is_main, leads["dep"], leads["free"]))

        for k, v in (("label", label), ("eligible", eligible), ("d0", d0), ("truth", truth),
                     ("f_dep", f_dep), ("f_free", f_free), ("hist", hist), ("trend", trend),
                     ("main", np.full(len(t), is_main)),
                     ):
            pooled[k].append(v)
        pooled.setdefault("active", []).append(active_pt)
        pooled.setdefault("level", []).append(np.full(len(t), level))

    P = {k: np.concatenate(v) for k, v in pooled.items()}
    m = P["main"]

    def pooled_mae(pred: np.ndarray) -> list[float]:
        sel = m & P["active"]
        return [round(float(np.abs(pred[sel, k] - P["truth"][sel, k]).mean()), 2) for k in range(horizon)]

    d0_rep = np.repeat(P["d0"][:, None], horizon, axis=1)
    summary = {
        "hold_out": [f"{hold_lo:%Y-%m-%d}", f"{hold_hi:%Y-%m-%d}"],
        "main_stations": [f"{r['clinic']}/{r['category']}" for r in rows if r["main"]],
        "forecast_mae_patients": {
            "horizons_min": [15 * (k + 1) for k in range(horizon)],
            "lstm_as_deployed": pooled_mae(P["f_dep"]),
            "lstm_leak_free": pooled_mae(P["f_free"]),
            "persistence": pooled_mae(d0_rep),
            "historical_profile": pooled_mae(P["hist"]),
        },
    }

    # early warning, pooled over main stations; scores are relative to each station's level
    el = m & P["eligible"]
    lab = P["label"][el]
    lvl = P["level"][el]
    scores = {
        "lstm_as_deployed": P["f_dep"].max(axis=1)[el] / lvl,
        "lstm_leak_free": P["f_free"].max(axis=1)[el] / lvl,
        "historical_profile": P["hist"].max(axis=1)[el] / lvl,
        "trend_extrapolation": P["trend"][el] / lvl,
        "persistence": P["d0"][el] / lvl,
    }
    budget = int((scores["lstm_leak_free"] >= 1).sum())
    ew = {"decision_points": int(el.sum()), "positives": int(lab.sum()),
          "base_rate": round(float(lab.mean()), 3), "alert_budget_for_baselines": budget, "methods": {}}
    for name, sc in scores.items():
        entry = {"average_precision": round(float(average_precision_score(lab, sc)), 3)}
        if name.startswith("lstm"):
            entry["at_congestion_level"] = prf(sc >= 1, lab)
        entry["at_same_alert_budget"] = prf(top_n_alerts(sc, budget), lab)
        ew["methods"][name] = entry
    summary["early_warning_decisions"] = ew

    main_eps = [e for e in episodes if e[1]]
    for key, idx in (("as_deployed", 2), ("leak_free", 3)):
        leads = [e[idx] for e in main_eps if e[idx] is not None]
        summary.setdefault("congestion_onsets", {})[key] = {
            "onsets": len(main_eps),
            "warned_in_advance": len(leads),
            "share_warned": round(len(leads) / len(main_eps), 3) if main_eps else None,
            "mean_lead_min": round(float(np.mean(leads)), 1) if leads else None,
            "median_lead_min": float(np.median(leads)) if leads else None,
            "share_warned_30min_plus": round(sum(x >= 30 for x in leads) / len(main_eps), 3) if main_eps else None,
        }

    print(json.dumps(summary, indent=1))
    print("\nPer station:")
    for r in sorted(rows, key=lambda r: -r["events"]):
        print(json.dumps(r))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "stations": rows}, indent=1))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
