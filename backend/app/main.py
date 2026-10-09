from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.services import topics
from app.services.reranker import get_reranker


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the cross-encoder at startup so the first request doesn't pay its ~8s load.
    get_reranker()
    # Backfill knowledge-base topics for documents ingested before topics existed.
    topics.ensure_topics_in_background()
    yield


app = FastAPI(title="MediQ AI Queue & Wait Time Advisor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
