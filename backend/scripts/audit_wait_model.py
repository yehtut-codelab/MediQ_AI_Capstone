"""Audit of the deployed XGBoost wait-time model (Models/waiting_time_xgb.pkl).

The training script for this model is not in the repository, so the audit
works from the artefact, its metadata and Clean_Dataset/cleaned_events.csv:

  1. Reproduce the reported validation numbers (and so recover how the
     `queue_depth` input was defined at training time).
  2. Check each input for information that is not available when a patient
     arrives, and measure how much the score depends on it.
  3. Score the model the way the app actually calls it.
  4. Compare with baselines stronger than the two in the metadata.
  5. Check calibration: bias, the ±MAE range shown in the UI, and the
     green / amber / red status bands.

Usage (from backend/):
    .venv/Scripts/python.exe scripts/audit_wait_model.py [--out results.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import livestate, models_service  # noqa: E402

RNG = np.random.default_rng(42)


def ns(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def depth_at_arrival(df: pd.DataFrame, keys: list[str], strict: bool, include_self: bool) -> np.ndarray:
    """Patients in the same group whose wait had started and whose service had
    not, at this patient's waiting_start. strict=True uses start < t < end
    (never counts the patient themself); strict=False uses start <= t < end."""
    out = np.zeros(len(df), dtype=float)
    pos = np.arange(len(df))
    ws_all, ss_all = ns(df["waiting_start"]), ns(df["service_start"])
    for _, idx in df.groupby(keys, sort=False).indices.items():
        ws, ss = ws_all[idx], ss_all[idx]
        ws_sorted, ss_sorted = np.sort(ws), np.sort(ss)
        if strict:
            d = np.searchsorted(ws_sorted, ws, "left") - np.searchsorted(ss_sorted, ws, "right")
        else:
            d = np.searchsorted(ws_sorted, ws, "right") - np.searchsorted(ss_sorted, ws, "right")
            if not include_self:
                d = d - (ss > ws)
        out[pos[idx]] = d
    return np.clip(out, 0, None)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"MAE": round(float(np.abs(err).mean()), 3),
            "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
            "R2": round(1 - ss_res / ss_tot, 3)}


def band(x: np.ndarray) -> np.ndarray:
    return np.where(x < 30, 0, np.where(x < 60, 1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = models_service.wait_meta()
    model = models_service._xgb()
    feats = meta["features"]
    lag_lo, lag_hi = meta["appointment_lag_clip"]

    df = pd.read_csv(
        livestate.CSV_PATH,
        usecols=["visit_id", "clinic_name", "service_category", "room_type", "room_number",
                 "service_date", "hour_of_day", "day_of_week", "month", "waiting_start",
                 "service_start", "waiting_time_min", "appointment_lag_min",
                 "visit_position", "visit_length"],
        parse_dates=["service_date", "waiting_start", "service_start"],
    ).sort_values("waiting_start").reset_index(drop=True)
    cutoff = pd.Timestamp(meta["train_cutoff"])
    is_val = (df["service_date"] > cutoff).to_numpy()
    y = df["waiting_time_min"].to_numpy(dtype=float)
    tr, va = df[~is_val], df[is_val]
    yv = y[is_val]
    out: dict = {"rows": {"train": int((~is_val).sum()), "validation": int(is_val.sum())},
                 "reported": meta["evaluation"] | {"baselines": meta["baselines"]}}

    base = pd.DataFrame({
        "cat_encoded": df["service_category"].map(meta["category_encoding"]).fillna(meta["category_encoding"]["other"]),
        "clinic_encoded": df["clinic_name"].map(meta["clinic_encoding"]),
        "hour_of_day": df["hour_of_day"], "day_of_week": df["day_of_week"], "month": df["month"],
        "visit_position": df["visit_position"], "visit_length": df["visit_length"],
        "appointment_lag_clipped": df["appointment_lag_min"].clip(lag_lo, lag_hi),
    })

    def predict(X: pd.DataFrame) -> np.ndarray:
        return np.expm1(model.predict(X[feats].to_numpy(dtype=float)))

    # ---- 1. reproduce: which queue_depth definition was the model trained on? ----
    day = "service_date"
    candidates = {
        "clinic+category, same day, start<t<end": depth_at_arrival(df, [day, "clinic_name", "service_category"], True, False),
        "clinic+category, same day, incl. self": depth_at_arrival(df, [day, "clinic_name", "service_category"], False, True),
        "category (clinics pooled), start<t<end": depth_at_arrival(df, [day, "service_category"], True, False),
        "category (clinics pooled), incl. self": depth_at_arrival(df, [day, "service_category"], False, True),
        "clinic (all categories), start<t<end": depth_at_arrival(df, [day, "clinic_name"], True, False),
        "room, start<t<end": depth_at_arrival(df, [day, "clinic_name", "room_type", "room_number"], True, False),
    }
    repro = {}
    for name, qd in candidates.items():
        X = base.assign(queue_depth=qd)
        repro[name] = metrics(yv, predict(X[is_val]))
    out["reproduction_by_queue_depth_definition"] = repro
    best = min(repro, key=lambda k: abs(repro[k]["MAE"] - meta["evaluation"]["val_MAE_min"]))
    out["closest_definition"] = best
    X_all = base.assign(queue_depth=candidates[best])
    Xv = X_all[is_val].reset_index(drop=True)
    p_ref = predict(Xv)
    ref = metrics(yv, p_ref)
    out["reference_validation"] = ref
    out["reported_baselines_reproduced"] = {
        "global_mean": metrics(yv, np.full(len(yv), y[~is_val].mean())),
        "category_mean": metrics(yv, va["service_category"].map(tr.groupby("service_category")["waiting_time_min"].mean()).to_numpy()),
    }

    # ---- 2. dependence on each input (permutation within validation) ----
    perm = {}
    for f in feats:
        Xp = Xv.copy()
        Xp[f] = RNG.permutation(Xp[f].to_numpy())
        perm[f] = round(metrics(yv, predict(Xp))["MAE"] - ref["MAE"], 3)
    out["mae_increase_when_input_shuffled"] = dict(sorted(perm.items(), key=lambda kv: -kv[1]))

    # visit_length is the TOTAL number of steps in the visit — only known after the visit ends
    typical = tr.groupby(["clinic_name", "service_category"])[["visit_position", "visit_length"]].median()
    typ = va[["clinic_name", "service_category"]].merge(typical, left_on=["clinic_name", "service_category"],
                                                        right_index=True, how="left").fillna({"visit_position": 1, "visit_length": 2})
    X_len = Xv.assign(visit_length=typ["visit_length"].to_numpy())
    X_len_floor = Xv.assign(visit_length=np.maximum(Xv["visit_position"], typ["visit_length"].to_numpy()))
    out["without_future_visit_length"] = {
        "typical_length_for_station": metrics(yv, predict(X_len)),
        "max(steps_so_far, typical_length)": metrics(yv, predict(X_len_floor)),
    }

    # ---- 3. the way the app calls it ----
    qd_app = candidates["clinic+category, same day, incl. self"][is_val]   # livestate.queue_depth: start <= t < end
    X_app_reg = Xv.assign(visit_position=typ["visit_position"].to_numpy(), visit_length=typ["visit_length"].to_numpy(),
                          appointment_lag_clipped=0.0, queue_depth=qd_app)
    X_app_lag = X_app_reg.assign(appointment_lag_clipped=Xv["appointment_lag_clipped"])
    out["as_called_by_the_app"] = {
        "typical position+length, lag=0, app queue depth": metrics(yv, predict(X_app_reg)),
        "same, but with the real appointment lag": metrics(yv, predict(X_app_lag)),
    }

    # ---- 4. stronger baselines (all fitted on the training period only) ----
    def group_median(keys: list[str]) -> np.ndarray:
        med = tr.groupby(keys)["waiting_time_min"].median().rename("m")
        fallback = va["service_category"].map(tr.groupby("service_category")["waiting_time_min"].median())
        m = va[keys].merge(med, left_on=keys, right_index=True, how="left")["m"]
        return m.fillna(fallback).to_numpy()

    # live baseline: median wait of the last 10 patients who STARTED service at this
    # station earlier the same day (observable at arrival time)
    rolling = np.full(len(va), np.nan)
    va_pos = np.arange(len(va))
    for _, idx in va.groupby(["service_date", "clinic_name", "service_category"], sort=False).indices.items():
        g = va.iloc[idx]
        order = np.argsort(ns(g["service_start"]), kind="stable")
        ss_sorted = ns(g["service_start"])[order]
        w_sorted = g["waiting_time_min"].to_numpy(dtype=float)[order]
        k = np.searchsorted(ss_sorted, ns(g["waiting_start"]), "right")
        rolling[va_pos[idx]] = [np.median(w_sorted[max(0, n - 10):n]) if n else np.nan for n in k]
    hour_med = group_median(["clinic_name", "service_category", "hour_of_day"])
    rolling_filled = np.where(np.isnan(rolling), hour_med, rolling)
    out["baselines_on_validation"] = {
        "category median": metrics(yv, group_median(["service_category"])),
        "clinic+category median": metrics(yv, group_median(["clinic_name", "service_category"])),
        "clinic+category+hour median": metrics(yv, hour_med),
        "clinic+category+hour+weekday median": metrics(yv, group_median(["clinic_name", "service_category", "hour_of_day", "day_of_week"])),
        "live: median of last 10 waits at the station": metrics(yv, rolling_filled),
        "xgboost (reference)": ref,
        "xgboost (as called by the app)": out["as_called_by_the_app"]["typical position+length, lag=0, app queue depth"],
    }

    # ---- 5. calibration of the reference predictions ----
    mae = ref["MAE"]
    b_true, b_pred = band(yv), band(p_ref)
    names = ["green <30", "amber 30-60", "red >60"]
    out["calibration"] = {
        "mean_actual_min": round(float(yv.mean()), 2), "mean_predicted_min": round(float(p_ref.mean()), 2),
        "median_actual_min": float(np.median(yv)), "median_predicted_min": round(float(np.median(p_ref)), 2),
        "share_within_plus_minus_MAE": round(float((np.abs(p_ref - yv) <= mae).mean()), 3),
        "share_actual_above_predicted_plus_MAE": round(float((yv > p_ref + mae).mean()), 3),
        "band_accuracy": round(float((b_true == b_pred).mean()), 3),
        "band_recall": {names[i]: round(float((b_pred[b_true == i] == i).mean()), 3) for i in range(3)},
        "band_share_actual": {names[i]: round(float((b_true == i).mean()), 3) for i in range(3)},
        "band_share_predicted": {names[i]: round(float((b_pred == i).mean()), 3) for i in range(3)},
        "max_prediction_min": round(float(p_ref.max()), 1),
        "mae_by_actual_wait": {
            lab: round(float(np.abs(p_ref[m] - yv[m]).mean()), 2)
            for lab, m in (("<30 min", yv < 30), ("30-60 min", (yv >= 30) & (yv < 60)), (">60 min", yv >= 60))
        },
    }

    per_cat = {}
    cat_med = group_median(["clinic_name", "service_category", "hour_of_day"])
    for c, m in va.reset_index(drop=True).groupby("service_category").indices.items():
        per_cat[c] = {"n": int(len(m)), "xgboost_MAE": round(float(np.abs(p_ref[m] - yv[m]).mean()), 2),
                      "station_hour_median_MAE": round(float(np.abs(cat_med[m] - yv[m]).mean()), 2)}
    out["per_category"] = dict(sorted(per_cat.items(), key=lambda kv: -kv[1]["n"]))

    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
