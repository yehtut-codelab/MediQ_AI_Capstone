# MediQ AI — Agentic Digital Twin for Eye Clinic Operations

MediQ AI models an eye clinic as a **digital twin**: historical patient-flow data drives a
live simulation of queue depth, congestion, and wait times for any clinic/category/timestamp,
and a set of **LangGraph** agents orchestrate trained ML models and retrieval-augmented
evidence into wait estimates, bottleneck alerts, and operational recommendations.

📄 Docs: [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) · [`docs/FRAMEWORK_DECISION.md`](docs/FRAMEWORK_DECISION.md)

## Architecture

**Digital twin layer** (`services/livestate.py`) replays the historical dataset as live
clinic state — queue depth, recent wait times, and bucketed features — for any requested
timestamp. A timestamp outside the trained dataset's date range is mapped to the nearest
same-weekday/time-of-day analog rather than rejected, so the twin always has something to
simulate against.

**Trained models** (`Models/`) provide the quantitative core:
- XGBoost regressor — wait-time estimate
- Per-category Gaussian HMM — congestion state (Low/Medium/High)
- LSTM — 15-minute-bucket queue-depth forecast

**RAG evidence layer** (Qdrant + OpenAI embeddings + local cross-encoder reranker) retrieves
the nearest historical visit events for a new arrival, giving every estimate a grounded,
inspectable evidence set alongside the model prediction. Embeddings use the same OpenAI
account/API key as the LLM phrasing steps and the SOP Q&A agent (`OPENAI_API_KEY` is
therefore required everywhere, not just for LLM phrasing — see Environment below).

**Agentic orchestration** — two LangGraph graphs:

- **`WaitEstimateGraph`** (`agents/graph.py`) — single-arrival RAG estimate:
  `parse_context → retrieve → check_evidence →(widen, up to 2×)→ rerank → compute_stats → synthesize`
  Retrieval filters widen automatically when too few comparable historical cases are found.

- **Supervisor multi-agent graph** (`agents/supervisor_graph.py`) — clinic-wide operational
  analysis, fanning out to four specialists in parallel, then a fifth agent that synthesizes
  their findings, before fusing everything:

  ```
                      ┌─ waiting_time_agent ──────┐
  supervisor ─ fan-out┼─ bottleneck_agent ─────────┼─ predictive_agent ─ result_fusion ─ critique
                      ├─ queue_depth_forecast_agent┤                          │
                      └─ resource_planning ────────┘                          ▼
                                                         decision_support ─ reporting ─ END
  ```

  Each of the first four specialists reads live twin state and calls a trained model
  (XGBoost/RAG, HMM, LSTM, allocation heuristics). `predictive_agent` doesn't call a model
  itself — it fans in after all four complete and analyzes their combined findings into a
  trend-adjusted wait projection, an uncertainty band, a near-term outlook, and an overall
  operational risk score. `critique` then validates cross-model agreement (e.g. XGBoost vs.
  RAG spread, HMM-vs-forecast consistency) and scores confidence before `decision_support`
  produces prioritized recommendations.

Both graphs optionally call GPT to phrase the final explanation/report in plain language —
every number in that text is computed upstream by code/models, and both graphs fall back to a
template summary if that specific call fails. This is now only an error-recovery fallback,
not a "run without a key" mode: `OPENAI_API_KEY` is required end-to-end because retrieval
itself calls the OpenAI embeddings API before either graph reaches the phrasing step.

- **SOP & Healthcare Q&A agent** (`agents/qa_graph.py`) — the one genuinely tool-calling LLM
  agent in the app: `agent ⇄ tools` (LangGraph ReAct loop). The LLM decides whether/what to
  search, can issue repeated searches with refined queries, and reasons over retrieved SOP
  passages rather than rephrasing pre-computed numbers. Retrieval hits a dedicated
  `mediq_sop_docs` Qdrant collection (same OpenAI embedder + local cross-encoder reranker as
  the wait-time RAG). Like the other two graphs, this one **requires** `OPENAI_API_KEY` —
  open-ended reasoning over free-text documents has no deterministic fallback.

## Quick Start — Phase 1 (Excel → Qdrant → nearest-event search)

### 1. Start Qdrant

```bash
docker compose up -d qdrant
# dashboard: http://localhost:6333/dashboard
```

### 2. Install backend deps

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Place the dataset

Copy the wait time dataset Excel file into `data/raw/` (gitignored).

### 4. Set your OpenAI API key

`OPENAI_API_KEY` must be set in `backend/.env` **before ingesting** — embeddings are computed
via the OpenAI API (no offline fallback). See Environment below.

### 5. Ingest

```bash
python scripts/ingest_waittime.py                 # full run (~237K events, calls OpenAI embeddings API)
python scripts/ingest_waittime.py --limit 5000    # smoke test
python scripts/ingest_waittime.py --recreate      # drop & rebuild collection (needed if you
                                                    # previously ingested with a different
                                                    # embedding model/dimension)
```

### 6. Test nearest-event search

```bash
python scripts/query_similar.py --clinic "Eye Centre" --service-type Consultation --hour 9 --dow 3
```

### 7. (Optional) Ingest SOP documents for the Q&A agent

```bash
mkdir -p ../data/sop_docs   # drop PDF/DOCX/TXT SOP documents here (gitignored)
python scripts/ingest_sop_docs.py
python scripts/ingest_sop_docs.py --recreate   # drop & rebuild the SOP collection
```

### 8. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

## Key endpoints

| Endpoint | Graph / model | Purpose |
|---|---|---|
| `POST /api/v1/arrivals/estimate` | `WaitEstimateGraph` | RAG wait estimate for a new arrival |
| `GET /api/v1/arrivals/samples` | — | Random sample historical arrivals (for demo/testing) |
| `POST /api/v1/queue/simulate` | trained models | Model-only queue simulation (twin replay) |
| `POST /api/v1/patients/register` | `WaitEstimateGraph` + trained models | Register a patient, triage, pathway, wait estimate |
| `GET /api/v1/patients` | — | List registered patients |
| `GET /api/v1/stations/overview` | — | Per-station historical stats + today's registrations |
| `POST /api/v1/agent/analyze` | Supervisor multi-agent graph (5 agents) | Full clinic-wide bottleneck/resource/forecast/predictive analysis |
| `GET /api/v1/agent/dataset-range` | — | Trained-model dataset date range + categories |
| `POST /api/v1/qa/ask` | SOP & Healthcare Q&A agent | Tool-calling agent answer over ingested SOP documents (requires `OPENAI_API_KEY`) |
| `GET /api/v1/documents` | — | List ingested SOP documents |
| `POST /api/v1/documents/upload` | — | Upload + ingest a SOP document (PDF/DOCX/TXT) |
| `POST /api/v1/documents/{id}/reingest` | — | Retry ingestion for a failed document |
| `DELETE /api/v1/documents/{id}` | — | Remove a SOP document and its chunks |
| `GET /api/v1/collection/status` | — | Qdrant collection health/status |

## Layout

```
backend/
  app/
    api/routes.py             # FastAPI endpoints
    agents/
      graph.py                 # LangGraph WaitEstimateGraph (RAG estimate)
      supervisor_graph.py       # LangGraph supervisor multi-agent graph (clinic analysis)
      qa_graph.py               # LangGraph tool-calling SOP & Healthcare Q&A agent
    services/
      livestate.py              # digital-twin state: queue depth, recent waits, forecasts
      models_service.py         # loads/serves XGBoost, HMM, LSTM
      embedder.py, qdrant_service.py, reranker.py   # wait-event RAG retrieval
      sop_service.py, sop_ingest.py   # SOP-document Qdrant collection (mediq_sop_docs)
      document_registry.py       # SOP document metadata (status, chunk count) for /documents
      preprocess.py, pathway.py, triage.py, stations.py, registry.py
    config.py, schemas.py, main.py
  scripts/
    ingest_waittime.py    # Excel → Qdrant
    ingest_sop_docs.py    # PDF/DOCX/TXT → Qdrant (mediq_sop_docs)
    mine_pathways.py      # mine empirical patient pathways from wait time export
    query_similar.py      # CLI nearest-event check
  run_backend.cmd         # Windows helper to launch the API
Models/                  # trained model artifacts (XGBoost, HMM, LSTM + metadata)
frontend/                # Next.js dashboard — arrival simulator, registration, station overview,
                          # agent console, SOP Q&A chat, document management
docs/                    # spec + ADRs
data/raw/                # wait-time dataset (gitignored)
data/sop_docs/           # SOP/procedure PDFs, DOCX, TXT for the Q&A agent (gitignored)
```

## Environment

`backend/.env` (copy from `.env.example`):

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=mediq_wait_events
DATA_FILE=../data/raw/TTSH Oct 25 - 04 May 26 - WaitTimeAdded.xlsx

# Required everywhere — embeddings (wait-event + SOP RAG search) and LLM phrasing/Q&A
# all call the OpenAI API. There is no local/offline fallback.
OPENAI_API_KEY=
LLM_MODEL=gpt-5.5                    # OpenAI model used wherever an LLM is called
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536                   # must match EMBEDDING_MODEL's output size —
                                      # changing either requires --recreate on both
                                      # Qdrant collections and a full re-ingest
CORS_ORIGINS=http://localhost:3000
QDRANT_SOP_COLLECTION=mediq_sop_docs
SOP_DOCS_DIR=../data/sop_docs
```
