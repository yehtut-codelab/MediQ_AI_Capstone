from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "mediq_wait_events"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    data_file: Path = Path("../data/raw/TTSH Oct 25 - 04 May 26 - WaitTimeAdded.xlsx")
    openai_api_key: str = ""
    llm_model: str = "gpt-5.5"
    cors_origins: str = "http://localhost:3000"

    # ingestion data-quality bounds (spec DQ-2)
    max_wait_sec: float = 14_400.0  # 4 hours
    min_wait_sec: float = 0.0

    # SOP & healthcare Q&A RAG
    qdrant_sop_collection: str = "mediq_sop_docs"
    sop_docs_dir: Path = Path("../data/sop_docs")
    sop_chunk_size: int = 1000
    sop_chunk_overlap: int = 200
    sop_retrieval_k: int = 15
    sop_rerank_top_n: int = 5
    # gpt-6-luna only emits reasoning summaries ("thinking") at high effort;
    # low/medium skip reasoning entirely on typical SOP questions.
    qa_reasoning_effort: str = "high"
    # Persona used when a request doesn't pick one (ids in app/agents/qa_personas.py).
    qa_default_persona: str = "clinical_staff"
    # Best cross-encoder score below this = the knowledge base doesn't cover the
    # question (calibrated: covered questions scored -3.5..6.5, off-topic <= -7.5).
    qa_covered_min_score: float = -5.0


settings = Settings()
