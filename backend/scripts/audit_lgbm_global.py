"""Audit of the global LightGBM waiting-time model before it replaces XGBoost.

Works from Models/3_Trained_Models/deployment_global_model and the pipeline's
feature table models_codes/data/cleaned_events.csv (both local-only):
  1. Reproduce the reported pooled test MAE with the inputs as trained.
  2. Re-score with only what the app knows when a patient is registered:
     arrival hour instead of service-start hour, no room, no visit history.
  3. Compare with simple medians fitted on everything before the test window.
  4. Write Models/lgbm_app_profile.json: the per-station defaults the app uses
     for inputs it does not have, and the accuracy measured with those defaults
     (shown on the app's model card).

Usage (from backend/):
  python -m scripts.audit_lgbm_global [--out scripts/backtest_results/lgbm_global_audit.json]
      [--no-profile]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "Models" / "3_Trained_Models" / "deployment_global_model"
DATA_PATH = ROOT / "models_codes" / "data" / "cleaned_events.csv"
PROFILE_PATH = ROOT / "Models" / "lgbm_app_profile.json"


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {"MAE": round(float(np.abs(err).mean()), 2),
            "RMSE": round(float(np.sqrt((err ** 2).mean())), 2),
            "R2": round(float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-profile", action="store_true", help="do not rewrite Models/lgbm_app_profile.json")
    args = ap.parse_args()

    meta = json.loads((MODEL_DIR / "deployment_metadata.json").read_text())
    calib = json.loads((MODEL_DIR / "quantile_calibration.json").read_text())["shift_min"]
    model = joblib.load(MODEL_DIR / "lgbm_global.pkl")
    quant = {q: joblib.load(MODEL_DIR / f"lgbm_global_{q}.pkl") for q in ("p10", "p50", "p90")}
    breach = joblib.load(MODEL_DIR / "breach_classifier.pkl")
    room = joblib.load(MODEL_DIR / "room_encoder.pkl")
    feats = list(model.feature_name_)

    df = pd.read_csv(DATA_PATH, parse_dates=["service_date", "waiting_start", "service_start"])
    df = df[df["wait_min"].notna()].reset_index(drop=True)
    # the global model was trained with the category as a pandas categorical
    df["service_cat"] = pd.Categorical(df["service_category"], categories=meta["categories"])
    df["cat_encoded"] = df["service_category"].map(meta["category_encoding"]).fillna(0).astype(int)
    df["clinic_encoded"] = df["clinic_name"].map(meta["clinic_encoding"])
    df["room_number_encoded"] = df["service_point"].map(room["encoding"]).fillna(room["fallback"])
    df["arrival_hour"] = df["waiting_start"].dt.hour
    df["arrival_dow"] = df["waiting_start"].dt.dayofweek
    df["arrival_month"] = df["waiting_start"].dt.month

    test_cutoff = pd.Timestamp(meta["split"]["test_cutoff"])
    is_test = df["service_date"] > test_cutoff
    past, te = df[~is_test], df[is_test]
    y = te["wait_min"].to_numpy(dtype=float)
    out: dict = {"rows": {"before_test": len(past), "test": len(te)},
                 "features": feats,
                 "hour_of_day_equals_service_start_hour": round(float((df["hour_of_day"] == df["service_start"].dt.hour).mean()), 4),
                 "hour_of_day_equals_arrival_hour": round(float((df["hour_of_day"] == df["arrival_hour"]).mean()), 4)}

    def predict(X: pd.DataFrame, m=model, raw: bool = False) -> np.ndarray:
        Xf = X[feats]
        num = [c for c in feats if c != "service_cat"]
        p = m.predict(Xf.assign(**{c: Xf[c].fillna(0) for c in num}))
        return np.clip(p if raw else np.expm1(p), 0, None)

    station = ["clinic_name", "service_category"]

    def typical(col: str, keys: list[str]) -> np.ndarray:
        """Median of `col` before the test window, per group, for the test rows."""
        t = past.groupby(keys)[col].median().rename("t")
        fallback = te["service_category"].map(past.groupby("service_category")[col].median())
        return te.join(t, on=keys)["t"].fillna(fallback).fillna(past[col].median()).to_numpy()

    arrival_clock = dict(hour_of_day=te["arrival_hour"], day_of_week=te["arrival_dow"], month=te["arrival_month"])
    no_room = dict(room_number_encoded=typical("room_number_encoded", station),
                   room_queue_depth=typical("room_queue_depth", station))
    no_history = dict(prior_visit_count=typical("prior_visit_count", station))
    no_position = dict(visit_position=typical("visit_position", station))
    no_appointment = dict(appointment_lag_clipped=typical("appointment_lag_clipped", station))

    # ---- 1 + 2. as trained, then with app-knowable inputs only ----
    variants = {
        "as trained (reference)": te,
        "arrival hour instead of service-start hour": te.assign(**arrival_clock),
        "typical room (room unknown)": te.assign(**no_room),
        "typical prior visits (history unknown)": te.assign(**no_history),
        "typical visit position": te.assign(**no_position),
        "app conditions: arrival hour, no room, no history": te.assign(**arrival_clock, **no_room, **no_history),
        "app conditions + typical visit position": te.assign(**arrival_clock, **no_room, **no_history, **no_position),
        "app conditions + typical visit position, no appointment time":
            te.assign(**arrival_clock, **no_room, **no_history, **no_position, **no_appointment),
    }
    APP = "app conditions + typical visit position"
    APP_NO_APPT = "app conditions + typical visit position, no appointment time"
    preds = {name: predict(X) for name, X in variants.items()}
    out["lightgbm"] = {name: metrics(y, p) for name, p in preds.items()}

    # ---- 3. simple medians fitted before the test window ----
    def group_median(frame_keys: list[str], test_keys: list[str]) -> np.ndarray:
        med = past.groupby(frame_keys)["wait_min"].median().rename("m")
        fallback = te["service_category"].map(past.groupby("service_category")["wait_min"].median())
        return te.join(med, on=test_keys)["m"].fillna(fallback).to_numpy()

    past = past.assign(h=past["arrival_hour"], d=past["arrival_dow"])
    te = te.assign(h=te["arrival_hour"], d=te["arrival_dow"])
    out["baselines"] = {
        "station median": metrics(y, group_median(station, station)),
        "station + arrival-hour median": metrics(y, group_median(station + ["h"], station + ["h"])),
        "station + weekday + arrival-hour median (Station Overview)":
            metrics(y, group_median(station + ["d", "h"], station + ["d", "h"])),
        "station + weekday + service-start-hour median":
            metrics(y, group_median(station + ["day_of_week", "hour_of_day"], station + ["day_of_week", "hour_of_day"])),
    }

    # ---- per category, as trained vs app conditions vs Station Overview median ----
    app = preds[APP]
    base = group_median(station + ["d", "h"], station + ["d", "h"])
    cats = te["service_category"].to_numpy()
    out["per_category"] = {
        c: {"n": int((cats == c).sum()),
            "as_trained_MAE": metrics(y[cats == c], preds["as trained (reference)"][cats == c])["MAE"],
            "app_conditions_MAE": metrics(y[cats == c], app[cats == c])["MAE"],
            "weekday_hour_median_MAE": metrics(y[cats == c], base[cats == c])["MAE"]}
        for c in meta["categories"] if (cats == c).any()
    }

    # ---- P10-P90 range under app conditions ----
    Xa = variants[APP]
    shift = {q: te["service_category"].map({c: v[q] for c, v in calib.items()}).fillna(0).to_numpy() for q in quant}
    for label, X in (("as trained", variants["as trained (reference)"]), ("app conditions", Xa)):
        q = np.sort(np.stack([np.clip(predict(X, quant[k], raw=True) + shift[k], 0, None) for k in ("p10", "p50", "p90")]), axis=0)
        out.setdefault("range", {})[label] = {
            "share_below_p10": round(float((y < q[0]).mean()), 3),
            "share_within_p10_p90": round(float(((y >= q[0]) & (y <= q[2])).mean()), 3),
            "share_above_p90": round(float((y > q[2]).mean()), 3),
            "median_width_min": round(float(np.median(q[2] - q[0])), 1),
            "p50_MAE": metrics(y, q[1])["MAE"],
        }

    # ---- breach classifier (wait over the clinical limit for the category) ----
    from sklearn.metrics import f1_score, roc_auc_score
    limits = meta["breach_thresholds_min"]
    limit = te["service_category"].map(limits).fillna(limits["default"]).to_numpy()
    is_breach = (y >= limit).astype(int)
    b_feats = list(breach.feature_name_)
    for label, X in (("as trained", variants["as trained (reference)"]), ("app conditions", Xa)):
        prob = breach.predict_proba(X[b_feats].fillna(0))[:, 1]
        flag = (prob >= meta["breach_decision_threshold"]).astype(int)
        out.setdefault("breach", {})[label] = {
            "breach_rate": round(float(is_breach.mean()), 3),
            "AUC": round(float(roc_auc_score(is_breach, prob)), 3),
            "F1": round(float(f1_score(is_breach, flag)), 3),
            "share_flagged": round(float(flag.mean()), 3),
            "precision": round(float(is_breach[flag == 1].mean()), 3) if flag.any() else None,
            "recall": round(float(flag[is_breach == 1].mean()), 3),
        }

    # ---- 4. what the app needs: defaults for missing inputs + measured accuracy ----
    cols = {"room_number_encoded": 3, "room_queue_depth": 1, "prior_visit_count": 1,
            "visit_position": 1, "appointment_lag_clipped": 1}
    by_station = past.groupby(station)[list(cols)].median()
    by_category = past.groupby("service_category")[list(cols)].median()
    defaults: dict = {"by_category": {c: {k: round(float(v), cols[k]) for k, v in row.items()}
                                      for c, row in by_category.iterrows()}}
    for (clinic, category), row in by_station.iterrows():
        defaults.setdefault(clinic, {})[category] = {k: round(float(v), cols[k]) for k, v in row.items()}
    val_cutoff = pd.Timestamp(meta["split"]["val_cutoff"])
    train = df[df["service_date"] <= val_cutoff]
    profile = {
        "generated_by": "backend/scripts/audit_lgbm_global.py",
        "train": {"events": len(train), "from": train["service_date"].min().date().isoformat(),
                  "to": train["service_date"].max().date().isoformat()},
        "test": {"events": len(te), "from": te["service_date"].min().date().isoformat(),
                 "to": te["service_date"].max().date().isoformat()},
        "defaults": defaults,
        "evaluation": {
            "as_trained": out["lightgbm"]["as trained (reference)"],
            "app": out["lightgbm"][APP],
            "app_no_appointment": out["lightgbm"][APP_NO_APPT],
        },
        "range": out["range"]["app conditions"],
        "breach": out["breach"]["app conditions"],
        "baselines": [
            {"label": "Station median", **out["baselines"]["station median"]},
            {"label": "Station, weekday and hour median",
             **out["baselines"]["station + weekday + arrival-hour median (Station Overview)"]},
        ],
        "per_category": out["per_category"],
    }

    print(json.dumps(out, indent=1))
    if not args.no_profile:
        PROFILE_PATH.write_text(json.dumps(profile, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
