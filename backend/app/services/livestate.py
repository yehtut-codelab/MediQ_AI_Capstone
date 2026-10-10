"""Reconstruct clinic state 'as of' any timestamp from Clean_Dataset/cleaned_events.csv.

The demo has no live feed, so specialist agents replay the cleaned dataset:
queue depth = patients whose wait has started but service hasn't, bucket series
feeds the LSTM, recent completed waits feed the HMM.
"""

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.models_service import lstm_meta, wait_meta

CSV_PATH = Path(__file__).resolve().parents[3] / "Clean_Dataset" / "cleaned_events.csv"

CATEGORIES = ["visual_acuity", "consultation", "diagnostic_scan", "biometry",
              "visual_field_test", "refraction", "laser_procedure", "treatment", "other"]


@lru_cache(maxsize=1)
def _events() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        usecols=["clinic_name", "service_category", "waiting_start", "service_start",
                 "waiting_time_min", "visit_position", "visit_length"],
        parse_dates=["waiting_start", "service_start"],
    )
    return df.sort_values("waiting_start")


def dataset_range() -> tuple[datetime, datetime]:
    df = _events()
    return df["waiting_start"].min().to_pydatetime(), df["waiting_start"].max().to_pydatetime()


def dataset_size() -> int:
    return len(_events())


def analog_timestamp(at: datetime) -> tuple[datetime, bool]:
    """Map `at` onto an equivalent in-range timestamp when it falls outside
    dataset_range(): same weekday + time-of-day, on the nearest matching date
    inside the range (anchored to whichever boundary `at` is closer to — the
    dataset's latest date for future timestamps, its earliest for past ones).

    The trained models can only reconstruct queue state from events that were
    actually recorded, so a genuinely out-of-range timestamp (e.g. "now" in
    2026) has no real data to replay. Rather than refuse outright, this finds
    the closest historical analog with the same weekly/daily pattern so the
    simulator can still produce an approximate 'pattern replay' — callers
    should label results computed this way accordingly.

    Returns (mapped_at, was_mapped) — was_mapped is False when `at` was
    already inside range, in which case mapped_at == at.
    """
    lo, hi = dataset_range()
    if lo <= at <= hi:
        return at, False
    if at > hi:
        anchor = hi
        shift = (anchor.weekday() - at.weekday()) % 7
        candidate = anchor - timedelta(days=shift)
    else:
        anchor = lo
        shift = (at.weekday() - anchor.weekday()) % 7
        candidate = anchor + timedelta(days=shift)
    mapped = candidate.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    mapped = min(max(mapped, lo), hi)
    return mapped, True


def queue_depth(clinic: str, category: str, at: datetime) -> int:
    """Patients waiting (wait started, service not yet started) at `at`."""
    df = _events()
    mask = (
        (df["clinic_name"] == clinic)
        & (df["service_category"] == category)
        & (df["waiting_start"] <= at)
        & (df["service_start"] > at)
    )
    return int(mask.sum())


def queue_depths_all(clinic: str, at: datetime) -> dict[str, int]:
    return {cat: queue_depth(clinic, cat, at) for cat in CATEGORIES}


def recent_waits(clinic: str, category: str, at: datetime,
                 window_hours: int = 3, max_n: int = 40) -> list[float]:
    """Waits of patients who started service in the last `window_hours` before `at`."""
    df = _events()
    mask = (
        (df["clinic_name"] == clinic)
        & (df["service_category"] == category)
        & (df["service_start"] > at - timedelta(hours=window_hours))
        & (df["service_start"] <= at)
    )
    return df.loc[mask, "waiting_time_min"].tail(max_n).tolist()


def lstm_bucket_features(clinic: str, category: str, at: datetime) -> np.ndarray | None:
    """(lookback_steps, 8) feature rows for the LSTM, metadata feature order:
    queue_depth_mean, n_arrivals, waiting_time_mean, hour_of_day, day_of_week,
    month, cat_encoded, is_weekend. Returns None if the window has no activity."""
    meta = lstm_meta()
    steps, bucket_min = meta["lookback_steps"], meta["bucket_minutes"]
    cat_enc = wait_meta()["category_encoding"].get(category, 8)

    df = _events()
    df = df[(df["clinic_name"] == clinic) & (df["service_category"] == category)]

    rows, any_activity = [], False
    for i in range(steps, 0, -1):
        b_end = at - timedelta(minutes=bucket_min * (i - 1))
        b_start = b_end - timedelta(minutes=bucket_min)
        in_bucket = df[(df["waiting_start"] >= b_start) & (df["waiting_start"] < b_end)]
        mid = b_start + timedelta(minutes=bucket_min / 2)
        depth = int(((df["waiting_start"] <= mid) & (df["service_start"] > mid)).sum())
        n_arr = len(in_bucket)
        if n_arr or depth:
            any_activity = True
        rows.append([
            depth,
            n_arr,
            float(in_bucket["waiting_time_min"].mean()) if n_arr else 0.0,
            b_start.hour,
            b_start.dayofweek if hasattr(b_start, "dayofweek") else b_start.weekday(),
            b_start.month,
            cat_enc,
            1 if b_start.weekday() >= 5 else 0,
        ])
    return np.array(rows, dtype=float) if any_activity else None


def typical_visit_profile(clinic: str, category: str) -> dict:
    """Median visit_position / visit_length for XGB defaults."""
    df = _events()
    sub = df[(df["clinic_name"] == clinic) & (df["service_category"] == category)]
    if sub.empty:
        return {"visit_position": 1, "visit_length": 2}
    return {
        "visit_position": int(sub["visit_position"].median()),
        "visit_length": int(sub["visit_length"].median()),
    }
