"""Qdrant collection management and search for SOP / healthcare-procedure chunks.

Separate collection from the wait-time events (`qdrant_service.py`) since the
schema is unrelated — chunked document text rather than structured visit events.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import settings

_NAMESPACE = uuid.UUID("2f6b1a8e-4c9d-4a7f-8e1b-9d3c5a6f7b8c")


def ensure_sop_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = settings.qdrant_sop_collection
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        client.create_payload_index(name, field_name="document_id",
                                    field_schema=PayloadSchemaType.KEYWORD)


def chunk_point_id(document_id: str, chunk_index: int) -> str:
    """Deterministic ID → idempotent re-ingestion, same pattern as qdrant_service.point_id."""
    return str(uuid.uuid5(_NAMESPACE, f"{document_id}|{chunk_index}"))


def upsert_chunks(client: QdrantClient, chunks: list[dict], vectors: list[list[float]],
                  batch_size: int = 256) -> int:
    """`chunks` items: {document_id, document_name, chunk_index, text}."""
    points = [
        PointStruct(id=chunk_point_id(c["document_id"], c["chunk_index"]), vector=vec, payload=c)
        for c, vec in zip(chunks, vectors)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(settings.qdrant_sop_collection, points[i : i + batch_size], wait=True)
    return len(points)


def search_sop(client: QdrantClient, query_vector: list[float], limit: int = 15) -> list[dict]:
    name = settings.qdrant_sop_collection
    if not client.collection_exists(name):
        return []
    hits = client.query_points(
        collection_name=name, query=query_vector, limit=limit, with_payload=True,
    ).points
    return [{"score": h.score, **h.payload} for h in hits]


def document_text(client: QdrantClient, document_id: str) -> str:
    """A document's indexed text, rebuilt from its chunks in order — works even
    when the source file is no longer on disk."""
    name = settings.qdrant_sop_collection
    if not client.collection_exists(name):
        return ""
    selector = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection_name=name, scroll_filter=selector, limit=256,
                                      offset=offset, with_payload=True, with_vectors=False)
        points.extend(batch)
        if offset is None:
            break
    points.sort(key=lambda p: p.payload.get("chunk_index", 0))
    return "\n".join(p.payload.get("text", "") for p in points)


def delete_document(client: QdrantClient, document_id: str) -> None:
    """Remove all chunks belonging to one document (e.g. before re-ingesting or on delete)."""
    name = settings.qdrant_sop_collection
    if not client.collection_exists(name):
        return
    client.delete(
        collection_name=name,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        ),
    )


def sop_collection_status(client: QdrantClient) -> dict:
    name = settings.qdrant_sop_collection
    if not client.collection_exists(name):
        return {"exists": False, "points": 0}
    info = client.get_collection(name)
    return {"exists": True, "points": info.points_count, "status": str(info.status)}
