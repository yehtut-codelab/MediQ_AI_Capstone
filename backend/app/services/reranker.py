"""Cross-encoder reranking of retrieved wait events (local, no API cost).

Vector search recalls a broad candidate pool; the cross-encoder scores each
(query, event) pair jointly and keeps only the strongest matches, so the
estimate is generated from the top-N most relevant historical cases.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.services.preprocess import DAY_NAMES, daypart

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Load from the local HF cache first — skips the Hub round-trip that slowed
    # every cold start; fall back to downloading on a fresh machine.
    try:
        return CrossEncoder(RERANK_MODEL, local_files_only=True)
    except OSError:
        return CrossEncoder(RERANK_MODEL)


def event_context_text(hit: dict) -> str:
    """Operational context of a historical event, without the wait outcome
    so the reranker scores contextual similarity, not the answer."""
    dow = DAY_NAMES[int(hit["day_of_week"])]
    return (
        f"{hit['service_type']} at {hit['service_point']}, {hit['clinic']}, "
        f"on {dow} at {int(hit['hour']):02d}:00, {daypart(int(hit['hour']))} "
        f"{'weekend' if hit['is_weekend'] else 'weekday'}."
    )


def rerank(query_text: str, hits: list[dict], top_n: int = 10) -> list[dict]:
    if not hits:
        return []
    scores = get_reranker().predict(
        [(query_text, event_context_text(h)) for h in hits]
    )
    ranked = sorted(
        ({**h, "rerank_score": float(s)} for h, s in zip(hits, scores)),
        key=lambda h: h["rerank_score"],
        reverse=True,
    )
    return ranked[:top_n]


def rerank_chunks(query_text: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """Cross-encoder rerank for arbitrary text chunks (e.g. SOP document retrieval)."""
    if not hits:
        return []
    scores = get_reranker().predict([(query_text, h["text"]) for h in hits])
    ranked = sorted(
        ({**h, "rerank_score": float(s)} for h, s in zip(hits, scores)),
        key=lambda h: h["rerank_score"],
        reverse=True,
    )
    return ranked[:top_n]
