"""Back-test of the RETRAINED queue-depth LSTM (models_codes/models), which is
not the one the app currently serves from Models/.

Same question as backtest_early_warning.py — can the forecast warn of
congestion before it happens? — but evaluated on the retrained model's own
definitions, exactly as models_codes/scripts/03_train_lstm.py builds them:

  * one series per (service date, service category), both clinics pooled
  * 15-minute buckets that contain at least one arrival
  * target = mean point-in-time queue depth of the bucket
  * TEST = the last two months, untouched by training or early stopping

Congestion level, historical profile and baselines use only data before the
test period. Two variants are run, as in the other script: "as-trained" (bucket
waiting-time feature uses final waits, i.e. future information for patients
still waiting) and "leak-free" (still-waiting patients contribute elapsed wait).

Usage (from backend/):
    .venv/Scripts/python.exe scripts/backtest_early_warning_retrained.py [--out results.json]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
MC = ROOT / "models_codes"
PCTL, MIN_LEVEL = 90, 5
MAIN_CATEGORY_EVENTS = 15_000
OPEN_MIN, CLOSE_MIN = 8 * 60, 17 * 60


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_lstm", MC / "scripts" / "03_train_lstm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prf(alert: np.ndarray, label: np.ndarray) -> dict:
    tp, n_alert, n_pos = int((alert & label).sum()), int(alert.sum()), int(label.sum())
    return {"alerts": n_alert,
            "precision": round(tp / n_alert, 3) if n_alert else None,
            "recall": round(tp / n_pos, 3) if n_pos else None}


def top_n_alerts(score: np.ndarray, n: int) -> np.ndarray:
    alert = np.zeros(len(score), dtype=bool)
    if n > 0:
        alert[np.argsort(-score, kind="stable")[:n]] = True
    return alert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tl = load_training_module()
    L, H, B = tl.LOOKBACK, tl.HORIZON, tl.BUCKET_MINUTES
    F = tl.FEATURES

    df = pd.read_csv(MC / "data" / "cleaned_events.csv", parse_dates=["service_date"],
                     usecols=["service_date", "service_category", "waiting_start", "service_start",
                              "wait_min", "queue_depth", "is_holiday"])
    df["waiting_start"] = pd.to_datetime(df["waiting_start"])
    df["service_start"] = pd.to_datetime(df["service_start"])
    df = df[df["waiting_start"].notna()]
    buckets = tl.build_bucket_series(df)
    buckets["tod"] = buckets["bucket_start"].dt.hour * 60 + buckets["bucket_start"].dt.minute

    max_date = buckets["service_date"].max()
    test_cutoff = max_date - pd.DateOffset(months=tl.TEST_MONTHS)
    past = buckets[buckets["service_date"] <= test_cutoff]
    test = buckets[buckets["service_date"] > test_cutoff]
    print(f"History : {buckets['service_date'].min():%Y-%m-%d} … {test_cutoff:%Y-%m-%d}")
    print(f"TEST    : {test['service_date'].min():%Y-%m-%d} … {max_date:%Y-%m-%d}")

    in_hours = past["tod"].between(OPEN_MIN, CLOSE_MIN)
    level = {c: max(MIN_LEVEL, float(np.percentile(g["queue_depth_mean"], PCTL)))
             for c, g in past[in_hours].groupby("service_category")}
    profile = past.groupby(["service_category", "day_of_week", "tod"])["queue_depth_mean"].mean().to_dict()
    cat_events = df.groupby("service_category").size().to_dict()

    model = tl.QueueLSTM()
    model.load_state_dict(torch.load(MC / "models" / "lstm_queue_forecast.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load(MC / "models" / "lstm_queue_scaler.pkl")
    d_lo, d_hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])

    test_events = {k: g for k, g in df[df["service_date"] > test_cutoff]
                   .assign(bucket_start=lambda x: x["waiting_start"].dt.floor(f"{B}min"))
                   .groupby(["service_date", "service_category"])}

    recs = []      # one per decision point
    X_as, X_free = [], []
    onsets = []    # (category, main, n_possible, key) — filled after forecasting
    for (day, cat), grp in test.groupby(["service_date", "service_category"], sort=False):
        grp = grp.sort_values("bucket_start").reset_index(drop=True)
        T = len(grp)
        if T < L + H or cat not in level:
            continue
        raw = grp[F].to_numpy(dtype=float)
        tgt = grp["queue_depth_mean"].to_numpy(dtype=float)
        starts = grp["bucket_start"].to_numpy()
        ev = test_events[(day, cat)]
        ev_b = ev["bucket_start"].to_numpy()
        ev_ws, ev_ss = ev["waiting_start"].to_numpy(), ev["service_start"].to_numpy()
        ev_wait = ev["wait_min"].to_numpy(dtype=float)
        by_bucket = {b: np.nonzero(ev_b == b)[0] for b in starts}
        dow = int(grp["day_of_week"].iloc[0])
        for s in range(T - L - H + 1):
            t_dec = starts[s + L - 1] + np.timedelta64(B, "m")
            tod_dec = int(grp["tod"].iloc[s + L - 1]) + B
            if not OPEN_MIN <= tod_dec <= CLOSE_MIN:
                continue
            x = raw[s:s + L].copy()
            x_free = x.copy()
            for r in range(L):
                idx = by_bucket[starts[s + r]]
                if len(idx):
                    elapsed = (t_dec - ev_ws[idx]) / np.timedelta64(1, "m")
                    still = ~(ev_ss[idx] <= t_dec)          # not yet served (or never)
                    seen = np.where(still, elapsed, ev_wait[idx])
                    x_free[r, 2] = np.nanmean(seen)
            X_as.append(x)
            X_free.append(x_free)
            fut = slice(s + L, s + L + H)
            recs.append({
                "cat": cat, "day": day, "s": s, "t_dec": t_dec,
                "d0": tgt[s + L - 1], "d_prev": tgt[s + L - 3],
                "truth": tgt[fut],
                "hist": np.array([profile.get((cat, dow, int(t)), tgt[s + L - 1]) for t in grp["tod"].iloc[fut]]),
                "fut_starts": starts[fut],
            })
        for j in range(1, T):
            tod_j = int(grp["tod"].iloc[j])
            if tgt[j] >= level[cat] and tgt[j - 1] < level[cat] and OPEN_MIN <= tod_j <= CLOSE_MIN:
                onsets.append({"cat": cat, "day": day, "j": j, "start": starts[j], "has_history": j >= L})

    def forecast(Xs: list) -> np.ndarray:
        X = np.stack(Xs)
        n = X.shape[0]
        scaled = scaler.transform(X.reshape(-1, len(F))).reshape(n, L, len(F)).astype(np.float32)
        with torch.no_grad():
            out = model(torch.tensor(scaled)).numpy()
        return np.clip(out * (d_hi - d_lo) + d_lo, 0, None)

    f_as, f_free = forecast(X_as), forecast(X_free)
    cat = np.array([r["cat"] for r in recs])
    d0 = np.array([r["d0"] for r in recs])
    trend = d0 + (d0 - np.array([r["d_prev"] for r in recs]))
    truth = np.stack([r["truth"] for r in recs])
    hist = np.stack([r["hist"] for r in recs])
    lvl = np.array([level[c] for c in cat])
    main = np.array([cat_events[c] >= MAIN_CATEGORY_EVENTS for c in cat])
    eligible = d0 < lvl
    label = truth.max(axis=1) >= lvl

    def mae(pred, sel):
        return [round(float(np.abs(pred[sel, k] - truth[sel, k]).mean()), 2) for k in range(H)]

    d0_rep = np.repeat(d0[:, None], H, axis=1)
    summary = {
        "test_period": [f"{test['service_date'].min():%Y-%m-%d}", f"{max_date:%Y-%m-%d}"],
        "main_categories": sorted(c for c in level if cat_events[c] >= MAIN_CATEGORY_EVENTS),
        "congestion_level_patients": {c: round(v, 1) for c, v in level.items()},
        "forecast_mae_patients": {
            "steps_ahead": list(range(1, H + 1)),
            "lstm_as_trained": mae(f_as, main), "lstm_leak_free": mae(f_free, main),
            "persistence": mae(d0_rep, main), "historical_profile": mae(hist, main),
            "all_categories_overall": {
                "lstm_as_trained": round(float(np.abs(f_as - truth).mean()), 3),
                "lstm_leak_free": round(float(np.abs(f_free - truth).mean()), 3),
                "persistence": round(float(np.abs(d0_rep - truth).mean()), 3),
                "historical_profile": round(float(np.abs(hist - truth).mean()), 3),
            },
        },
    }

    el = main & eligible
    lab = label[el]
    scores = {
        "lstm_as_trained": f_as.max(axis=1)[el] / lvl[el],
        "lstm_leak_free": f_free.max(axis=1)[el] / lvl[el],
        "historical_profile": hist.max(axis=1)[el] / lvl[el],
        "trend_extrapolation": trend[el] / lvl[el],
        "persistence": d0[el] / lvl[el],
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

    # how early was each congestion onset flagged?
    index = {(r["day"], r["cat"], r["s"]): i for i, r in enumerate(recs)}
    main_cats = set(summary["main_categories"])
    onset_out = {}
    for key, fc in (("as_trained", f_as), ("leak_free", f_free)):
        leads, n_all, n_hist = [], 0, 0
        for o in onsets:
            if o["cat"] not in main_cats:
                continue
            n_all += 1
            n_hist += o["has_history"]
            lead = None
            for k in range(H - 1, -1, -1):                     # earliest decision first
                i = index.get((o["day"], o["cat"], o["j"] - L - k))
                if i is not None and eligible[i] and fc[i].max() >= lvl[i]:
                    lead = float((o["start"] - recs[i]["t_dec"]) / np.timedelta64(1, "m"))
                    break
            if lead is not None:
                leads.append(lead)
        onset_out[key] = {
            "onsets": n_all, "onsets_with_2h_history": int(n_hist), "warned_in_advance": len(leads),
            "share_of_all_onsets": round(len(leads) / n_all, 3) if n_all else None,
            "share_of_onsets_with_history": round(len(leads) / n_hist, 3) if n_hist else None,
            "mean_lead_min": round(float(np.mean(leads)), 1) if leads else None,
            "median_lead_min": float(np.median(leads)) if leads else None,
            "warned_30min_plus_share_of_all": round(sum(x >= 30 for x in leads) / n_all, 3) if n_all else None,
        }
    summary["congestion_onsets"] = onset_out

    per_cat = []
    for c in sorted(level, key=lambda c: -cat_events[c]):
        sel = cat == c
        if not sel.any():
            continue
        e = sel & eligible
        per_cat.append({
            "category": c, "events": int(cat_events[c]), "level": round(level[c], 1),
            "decision_points": int(e.sum()), "onsets_within_horizon": int((e & label).sum()),
            "mae_lstm_leak_free": mae(f_free, sel), "mae_persistence": mae(d0_rep, sel),
            "mae_hist_profile": mae(hist, sel),
            "alert_leak_free": prf((f_free.max(axis=1) >= lvl)[e], label[e]),
        })

    print(json.dumps(summary, indent=1))
    print("\nPer category:")
    for r in per_cat:
        print(json.dumps(r))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "categories": per_cat}, indent=1))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
