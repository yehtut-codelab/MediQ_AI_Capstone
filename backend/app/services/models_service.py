"""Loaders and predict functions for the trained models in Models/.

- 3_Trained_Models/deployment_global_model/
                            global LightGBM waiting-time model (target log1p(wait_min)),
                            P10/P50/P90 quantile models and the breach classifier
- lgbm_app_profile.json     per-station defaults for inputs the app does not have and
                            the accuracy measured with them (scripts/audit_lgbm_global.py)
- waiting_time_xgb.pkl      legacy XGBRegressor, kept only for scripts/audit_wait_model.py
- lstm_queue_forecast.pt    2-layer LSTM state dict, 15-min queue depth forecast
- clinic_hmms.pkl           per-category GaussianHMM congestion states (joblib)
"""

import json
import pickle
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

MODELS_DIR = Path(__file__).resolve().parents[3] / "Models"
WAIT_MODEL_DIR = MODELS_DIR / "3_Trained_Models" / "deployment_global_model"


# ── metadata ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def wait_meta() -> dict:
    return json.loads((WAIT_MODEL_DIR / "deployment_metadata.json").read_text())


@lru_cache(maxsize=1)
def wait_profile() -> dict:
    return json.loads((MODELS_DIR / "lgbm_app_profile.json").read_text())


@lru_cache(maxsize=1)
def xgb_meta() -> dict:
    return json.loads((MODELS_DIR / "waiting_time_metadata.json").read_text())


@lru_cache(maxsize=1)
def lstm_meta() -> dict:
    return json.loads((MODELS_DIR / "lstm_queue_metadata.json").read_text())


@lru_cache(maxsize=1)
def hmm_meta() -> dict:
    return json.loads((MODELS_DIR / "hmm_metadata.json").read_text())


# ── LightGBM waiting time ────────────────────────────────────────────────

APPOINTMENT_LAG_CLIP = (-180, 60)  # minutes, as clipped by the training pipeline


@lru_cache(maxsize=1)
def _xgb():
    """Legacy model, superseded by the LightGBM model below."""
    with open(MODELS_DIR / "waiting_time_xgb.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _wait_models() -> dict:
    return {
        "point": joblib.load(WAIT_MODEL_DIR / "lgbm_global.pkl"),
        "quantiles": {q: joblib.load(WAIT_MODEL_DIR / f"lgbm_global_{q}.pkl")
                      for q in ("p10", "p50", "p90")},
        "breach": joblib.load(WAIT_MODEL_DIR / "breach_classifier.pkl"),
        "calibration": json.loads((WAIT_MODEL_DIR / "quantile_calibration.json").read_text())["shift_min"],
    }


def wait_defaults(clinic: str, category: str) -> dict:
    """Typical values for the inputs the app cannot supply (room, visit
    history, visit position, and the appointment lag when no appointment is
    given), per station with a per-category fallback."""
    defaults = wait_profile()["defaults"]
    return (defaults.get(clinic, {}).get(category)
            or defaults["by_category"].get(category)
            or defaults["by_category"]["other"])


def wait_error_min(has_appointment: bool) -> float:
    """Hold-out MAE of the model as the app calls it (typical room, no visit
    history), with or without a real appointment time."""
    evaluation = wait_profile()["evaluation"]
    return evaluation["app" if has_appointment else "app_no_appointment"]["MAE"]


def predict_wait(category: str, features: dict) -> dict:
    """Point estimate, calibrated P10-P90 range and breach flag for one patient.
    `features` holds every numeric model input (see livestate.wait_model_features)."""
    meta, m = wait_meta(), _wait_models()
    if category not in meta["category_encoding"]:
        category = "other"
    # The regressors take the category as a pandas categorical; the breach
    # classifier takes its integer code.
    X = pd.DataFrame([features]).assign(
        service_cat=pd.Categorical([category], categories=meta["categories"]),
        cat_encoded=meta["category_encoding"][category],
    )
    point = m["point"]
    estimate = max(0.0, float(np.expm1(point.predict(X[list(point.feature_name_)])[0])))  # log1p / expm1

    shift = m["calibration"].get(category, {})
    p10, p50, p90 = sorted(
        max(0.0, float(qm.predict(X[list(qm.feature_name_)])[0]) + shift.get(q, 0.0))
        for q, qm in m["quantiles"].items()
    )

    limits = meta["breach_thresholds_min"]
    breach = m["breach"]
    probability = float(breach.predict_proba(X[list(breach.feature_name_)])[0, 1])
    return {
        "estimate_min": estimate,
        "p10_min": p10, "p50_min": p50, "p90_min": p90,
        "breach_limit_min": limits.get(category, limits["default"]),
        "breach_probability": probability,
        "breach_flag": probability >= meta["breach_decision_threshold"],
    }


# ── LSTM queue depth forecast ────────────────────────────────────────────

class _QueueLSTM(nn.Module):
    """Architecture reconstructed from the checkpoint state dict."""

    def __init__(self, n_features: int = 8, hidden1: int = 64, hidden2: int = 32,
                 horizon: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        return self.linear(out[:, -1, :])


@lru_cache(maxsize=1)
def _lstm() -> tuple[_QueueLSTM, object]:
    meta = lstm_meta()
    model = _QueueLSTM(
        n_features=meta["n_features"],
        hidden1=meta["architecture"]["hidden1"],
        hidden2=meta["architecture"]["hidden2"],
        horizon=meta["horizon_steps"],
        dropout=meta["architecture"]["dropout"],
    )
    model.load_state_dict(
        torch.load(MODELS_DIR / "lstm_queue_forecast.pt", map_location="cpu",
                   weights_only=False)
    )
    model.eval()
    scaler = joblib.load(MODELS_DIR / "lstm_queue_scaler.pkl")
    return model, scaler


def forecast_queue_depth(bucket_features: np.ndarray) -> list[float]:
    """bucket_features: (lookback_steps, 8) raw feature rows in metadata order.
    Returns queue depth forecast for the next horizon_steps 15-min buckets."""
    model, scaler = _lstm()
    scaled = scaler.transform(bucket_features)
    with torch.no_grad():
        out = model(torch.tensor(scaled[None], dtype=torch.float32))[0].numpy()
    # invert MinMax scaling using target column 0 (queue_depth_mean)
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    return [max(0.0, round(float(v) * (hi - lo) + lo, 2)) for v in out]


# ── HMM congestion states ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _hmms() -> dict:
    return joblib.load(MODELS_DIR / "clinic_hmms.pkl")


def congestion_state(category: str, recent_waits_min: list[float]) -> dict:
    """Most likely current congestion state given recent observed waits."""
    entry = _hmms().get(category)
    if entry is None or not recent_waits_min:
        return {"category": category, "state": "unknown", "state_mean_wait": None}
    obs = np.array(recent_waits_min, dtype=float).reshape(-1, 1)
    states = entry["model"].predict(obs)
    current = int(states[-1])
    label = entry["state_map"][current]
    return {
        "category": category,
        "state": label,
        "state_mean_wait": round(float(entry["means"][current]), 1),
        "n_observations": len(recent_waits_min),
    }
