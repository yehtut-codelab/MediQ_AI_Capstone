"""Per-station aggregates.

station_overview() reads the Qdrant index (stations = service types). One full
payload scroll of the collection can take a few seconds, so results are cached
in-process; the historical index is append-only, so staleness only matters
after a re-ingestion — restart or wait out the TTL.

weekly_profile() reads Clean_Dataset/cleaned_events.csv (stations = the
trained models' service categories), because a weekday x hour split needs far
more events per cell than the Qdrant index holds.
"""

import statistics
import time
from functools import lru_cache

import pandas as pd
from qdrant_client import QdrantClient

from app.config import settings
from app.services import livestate
from app.services.models_service import wait_meta

CACHE_TTL_SEC = 3600
MIN_EVENTS = 200  # hide micro-stations (e.g. GVF n=4) from the dashboard
MIN_DAY_EVENTS = 50  # a weekday with fewer events gets no profile for that station
MIN_HOUR_EVENTS = 20  # an hour with fewer events gets no bar

_cache: dict = {"at": 0.0, "data": None}


def _band(median: float) -> str:
    if median < 30:
        return "green"
    if median < 60:
        return "amber"
    return "red"


def _percentile(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _scroll_all(client: QdrantClient) -> list[dict]:
    points: list[dict] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10_000,
            offset=offset,
            with_payload=["clinic", "service_type", "wait_min", "hour"],
            with_vectors=False,
        )
        points.extend(p.payload for p in batch)
        if offset is None:
            return points


def station_overview(client: QdrantClient) -> dict[str, list[dict]]:
    """{clinic: [station dicts sorted by median wait desc]}, cached."""
    if _cache["data"] is not None and time.time() - _cache["at"] < CACHE_TTL_SEC:
        return _cache["data"]

    groups: dict[tuple[str, str], list[dict]] = {}
    for p in _scroll_all(client):
        groups.setdefault((p["clinic"], p["service_type"]), []).append(p)

    result: dict[str, list[dict]] = {}
    for (clinic, service_type), events in groups.items():
        if len(events) < MIN_EVENTS:
            continue
        waits = sorted(e["wait_min"] for e in events)
        by_hour: dict[int, list[float]] = {}
        for e in events:
            by_hour.setdefault(e["hour"], []).append(e["wait_min"])
        median = round(statistics.median(waits), 1)
        result.setdefault(clinic, []).append({
            "service_type": service_type,
            "count": len(events),
            "median_wait_min": median,
            "p75_wait_min": round(_percentile(waits, 0.75), 1),
            "p90_wait_min": round(_percentile(waits, 0.90), 1),
            "band": _band(median),
            "hourly": [
                {"hour": h, "median_wait_min": round(statistics.median(v), 1), "count": len(v)}
                for h, v in sorted(by_hour.items())
                if 7 <= h <= 18 and len(v) >= MIN_HOUR_EVENTS
            ],
        })

    for clinic in result:
        result[clinic].sort(key=lambda s: s["median_wait_min"], reverse=True)

    _cache.update(at=time.time(), data=result)
    return result


def _profile(events: pd.DataFrame) -> dict:
    """Wait statistics for one station on one weekday (or all days pooled)."""
    waits = events["waiting_time_min"]
    median, p75, p90 = (round(float(v), 1) for v in waits.quantile([0.5, 0.75, 0.9]))
    hourly = []
    for hour, w in events[events["hour"].between(7, 18)].groupby("hour")["waiting_time_min"]:
        if len(w) < MIN_HOUR_EVENTS:
            continue
        h_median, h_p75, h_p90 = (round(float(v), 1) for v in w.quantile([0.5, 0.75, 0.9]))
        hourly.append({"hour": int(hour), "median_wait_min": h_median, "p75_wait_min": h_p75,
                       "p90_wait_min": h_p90, "count": len(w)})
    return {
        "count": len(events),
        "days": int(events["date"].nunique()),
        "median_wait_min": median,
        "p75_wait_min": p75,
        "p90_wait_min": p90,
        "band": _band(median),
        "hourly": hourly,
    }


def _holdout_accuracy(df: pd.DataFrame) -> dict:
    """How well the page's own method predicts an individual wait: medians per
    clinic, category, weekday and arrival hour are fitted on events up to the
    wait model's training cut-off and scored on the events after it."""
    cutoff = pd.Timestamp(wait_meta()["split"]["test_cutoff"])
    train, val = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    keys = ["clinic_name", "service_category", "dow", "hour"]
    cell = train.groupby(keys)["waiting_time_min"].median().rename("pred")
    fallback = val["service_category"].map(train.groupby("service_category")["waiting_time_min"].median())
    pred = val.join(cell, on=keys)["pred"].fillna(fallback)
    actual = val["waiting_time_min"]
    err = pred - actual
    return {
        "train_cutoff": cutoff.date().isoformat(),
        "train_events": len(train),
        "validation_events": len(val),
        "mae_min": round(float(err.abs().mean()), 1),
        "rmse_min": round(float((err ** 2).mean() ** 0.5), 1),
        "r2": round(float(1 - (err ** 2).sum() / ((actual - actual.mean()) ** 2).sum()), 2),
    }


@lru_cache(maxsize=1)
def weekly_profile() -> dict:
    """Historical wait pattern per station, split by day of week.

    Each station carries `profiles`: "all" (every day pooled) plus one entry
    per weekday ("0" = Monday) that has at least MIN_DAY_EVENTS events. The
    dataset is static, so the result is computed once per process.
    """
    df = livestate._events()[["clinic_name", "service_category", "waiting_start",
                              "waiting_time_min"]].dropna()
    start = df["waiting_start"]
    df = df.assign(dow=start.dt.dayofweek, hour=start.dt.hour, date=start.dt.normalize())

    clinics: dict[str, list[dict]] = {}
    for (clinic, category), events in df.groupby(["clinic_name", "service_category"]):
        if len(events) < MIN_EVENTS:
            continue
        profiles = {"all": _profile(events)}
        for dow, day_events in events.groupby("dow"):
            if len(day_events) >= MIN_DAY_EVENTS:
                profiles[str(dow)] = _profile(day_events)
        clinics.setdefault(clinic, []).append(
            {"category": category, "count": len(events), "profiles": profiles})

    for station_list in clinics.values():
        station_list.sort(key=lambda s: s["profiles"]["all"]["median_wait_min"], reverse=True)

    return {
        "date_from": start.min().date().isoformat(),
        "date_to": start.max().date().isoformat(),
        "total_events": sum(s["count"] for lst in clinics.values() for s in lst),
        "weekdays": sorted(int(d) for d in df["dow"].unique()),
        "accuracy": _holdout_accuracy(df),
        "clinics": clinics,
    }
