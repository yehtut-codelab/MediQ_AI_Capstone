"""Per-station (service type) aggregates computed from the Qdrant index.

One full payload scroll of the collection can take a few seconds, so results are
cached in-process; the historical index is append-only, so staleness only
matters after a re-ingestion — restart or wait out the TTL.
"""

import statistics
import time

from qdrant_client import QdrantClient

from app.config import settings

CACHE_TTL_SEC = 3600
MIN_EVENTS = 200  # hide micro-stations (e.g. GVF n=4) from the dashboard

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
                if 7 <= h <= 18 and len(v) >= 20
            ],
        })

    for clinic in result:
        result[clinic].sort(key=lambda s: s["median_wait_min"], reverse=True)

    _cache.update(at=time.time(), data=result)
    return result
