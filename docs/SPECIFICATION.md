# TTSH Eye Clinic — Agentic Queue & Wait Time Advisor (MediQ AI)
## System Specification v1.2

**Project**: NUS-ISS Capstone · Team 2
**Date**: 2026-07-08 (v1.0) · **Last updated**: 2026-10-11 (v1.2, brought in line with the built system)
**Companion documents**: `FRAMEWORK_DECISION.md` (ADR-001), `PATHWAY_TRIAGE.md` (triage and
pathway prediction), the repository `README.md` (setup and full endpoint list).
The dashboard's original visual reference, `ttsh_eye_clinic_digital_twin_v3_2.html`, is kept
locally and is not in the repository.

---

## 1. Problem Statement

TTSH Eye Centre patients experience unpredictable waits across a multi-station journey
(Registration → Pre-screening/VA → Dilation → Visual Field → OCT → Consultation → Pharmacy).
When a new patient arrives, staff have no data-driven estimate of that patient's expected
wait at each station.

**Goal**: on patient arrival, produce an evidence-based wait time and queue estimate from
trained models and from the most similar historical visit events in a vector database
(Qdrant), with an agentic reasoning layer that explains the estimate and analyses the
clinic as a whole.

## 2. Source Data

The system uses two datasets. Neither is committed (see NFR-3).

### 2.1 Raw wait-time export — RAG evidence and pathway mining

`TTSH Oct 25 - 04 May 26 - WaitTimeAdded.xlsx` — 237,190 station-visit events,
2025-10-01 → 2026-05-04, single sheet. It is embedded into Qdrant (§3.1) and mined for
patient pathways (`PATHWAY_TRIAGE.md`).

| Column | Type | Notes |
|---|---|---|
| `Anonymized_ID` | int | Patient pseudo-ID (repeats across visits/stations) |
| `Appointment Date Time` | datetime | Scheduled slot |
| `Registration Date Time` | datetime | 4,389 nulls |
| `Service Point` | str | Room/station, e.g. `EYE-VA Room 51`, `C1A-Consultation Room 26` (2,451 nulls) |
| `Service Type` | str | Consultation, OCT, VA, HVF, Orthoptic, … (6,329 nulls) |
| `Clinic Name` | str | `TTSH Eye Centre` (128,658) / `Clinic 1A` (108,532) |
| `Wait Start/End Date Time` | datetime | ~4.4–5.6K nulls |
| `Service Start/End Date Time` | datetime | ~5.7–5.9K nulls |
| `Total Contact Time in sec` | float | Service duration |
| `Waiting Time in sec` | float | **Contains corrupt negatives (min ≈ −6.6×10⁷ s)** |

### 2.2 Cleaned events — trained models and the digital twin

`Clean_Dataset/cleaned_events.csv` — 405,600 station-visit events, 2024-10-01 → 2025-09-30,
with a `service_category` column (visual_acuity, consultation, diagnostic_scan, biometry,
visual_field_test, refraction, laser_procedure, treatment, other). The LightGBM, HMM and
LSTM models were trained on it, and the digital twin replays it as clinic state (§6.1).

The two datasets cover different periods. The RAG evidence comes from 2025-10 onwards, the
models and the twin from the year before. This mismatch is not resolved (see §7).

### 2.3 Data-quality rules (applied when ingesting the raw export)

- **DQ-1** Drop rows with null `Wait Start Date Time` or null `Waiting Time in sec`.
- **DQ-2** Drop rows where `Waiting Time in sec` < 0 or > 4 h (14,400 s) — timestamp corruption.
- **DQ-3** Drop rows with null `Service Type` (cannot canonicalize the event).
- **DQ-4** Normalize `Service Point` naming variants (`EYE-VA Room 51` ≡ `VA Room 51`).
- **DQ-5** All datetimes localized to `Asia/Singapore`.

## 3. Architecture — arrival estimate (RAG path)

```
┌───────────┐   ①ingest    ┌──────────────┐
│ Excel     │─────────────▶│ Preprocess    │  pandas: DQ rules, station
│ dataset   │              │ + Canonicalize│  normalization, time features
└───────────┘              └──────┬───────┘
                                  │ event text + payload
                           ┌──────▼───────┐
                           │ Embedder      │  OpenAI text-embedding-3-small
                           │ (OpenAI API)  │  1536-dim
                           └──────┬───────┘
                                  │ vectors
┌───────────┐   ②arrival   ┌──────▼───────┐
│ New       │─────────────▶│ Qdrant        │  wait-event collection
│ patient   │   (FastAPI)  │ (Docker)      │  payload-filtered kNN
└───────────┘              └──────┬───────┘
                                  │ top-50 similar events
                           ┌──────▼────────────────────────┐
                           │ LangGraph agent                │
                           │ parse → retrieve → widen?      │
                           │ → rerank → statistics          │
                           │ → LLM synthesis                │
                           └──────┬────────────────────────┘
                                  │ estimate + explanation
                           ┌──────▼───────┐
                           │ Next.js       │
                           │ dashboard     │
                           └──────────────┘
```

### 3.1 Vectorization strategy

Each **row = one station-visit event**, embedded from a canonical sentence
(`preprocess.event_sentence`):

> `"Consultation at Consultation Room 26, Clinic 1A, on Thursday at 09:00, month 10,
> mid-morning weekday; waited 28 minutes, service took 8 minutes."`

Structured fields stored as Qdrant **payload** for pre-filtering:

```json
{
  "patient_id": 300000133,
  "clinic": "Clinic 1A",
  "service_type": "Consultation",
  "service_point": "Consultation Room 26",
  "hour": 9, "day_of_week": 3, "is_weekend": 0, "is_public_holiday": 0,
  "month": 10, "wait_min": 28.4, "contact_min": 8.4,
  "wait_start_iso": "2025-10-30T09:12:00+08:00"
}
```

**Retrieval = hybrid**: a Qdrant filter (`clinic`, `service_type`, `hour ± 2`,
`day_of_week` or weekend class) narrows the candidate set; cosine kNN on the embedding
ranks within it; a local cross-encoder then reranks the pool (§3.2).

- Collection: set by `QDRANT_COLLECTION` (default `mediq_wait_events`) · distance: Cosine ·
  dim: 1536 (`EMBEDDING_DIM`, must match the embedding model)
- Embeddings are computed through the OpenAI API, so ingestion and every query need
  `OPENAI_API_KEY`. v1.0 specified a local 384-dim MiniLM embedder; that was replaced.
- Deterministic point IDs: UUID5 of (`patient_id`, wait start, `service_point`) → idempotent re-ingestion
- A second collection, `mediq_sop_docs`, holds chunks of SOP documents for the Q&A agent (§6.3).

### 3.2 LangGraph agent (see ADR-001)

State machine `WaitEstimateGraph` (`backend/app/agents/graph.py`):

1. **parse_context** — build the query text from the operational context (service type,
   clinic, weekday, hour) plus the patient's presentation when given (diagnosis, medical
   history, current issue), and embed it.
2. **retrieve** — filtered kNN, k=50.
3. **check_evidence** *(conditional edge)* — if < 20 hits, widen the filter and loop back,
   max 2 widenings: first exact weekday → weekday/weekend class, then no time filter at all.
4. **rerank** — cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, runs locally)
   reranks the pool; the top 10 cases are kept.
5. **compute_stats** — median / P10 / P75 / P90 wait of those cases, and a confidence
   level: high (≥ 40 retrieved, no widening), medium (≥ 20), otherwise low.
6. **synthesize** — the LLM rewrites a template sentence into a patient-facing message and
   a staff-facing sentence. Every number is computed in code, not by the LLM; if the LLM
   call fails, the template text is returned.

### 3.3 API (FastAPI)

The table lists the endpoints this section describes. The full list is in the README.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/arrivals/estimate` | New patient arrival → RAG wait estimate |
| `POST` | `/api/v1/queue/simulate` | Model-based queue state for a clinic/category/timestamp (§6.1) |
| `POST` | `/api/v1/patients/register` | Register a patient: triage, pathway, model + RAG estimate |
| `GET` | `/api/v1/stations/overview` | Per-station historical stats + today's registrations |
| `GET` | `/api/v1/stations/weekly` | Per-station wait pattern by weekday and hour |
| `GET` | `/api/v1/collection/status` | Qdrant collection health / point count |
| `GET` | `/health` | Liveness |

`POST /arrivals/estimate` request (`diagnosis`, `medical_history`, `current_issue` and
`appointment_datetime` are optional):
```json
{
  "clinic": "TTSH Eye Centre",
  "service_type": "Consultation",
  "arrival_datetime": "2026-07-08T09:15:00+08:00",
  "appointment_datetime": "2026-07-08T09:30:00+08:00",
  "current_issue": "blurred vision for two weeks"
}
```
Response:
```json
{
  "estimated_wait_min": 27.5,
  "range_min": [9.0, 62.0],
  "confidence": "high",
  "evidence_count": 10,
  "filters_widened": 0,
  "similar_events_sample": [ ... top 10 ... ],
  "explanation": "Based on the top 10 most similar historical cases ...",
  "status_band": "amber"
}
```
`estimated_wait_min` is the median and `range_min` the P10–P90 of the reranked cases.
`status_band`: green < 30 min, amber 30–60, red > 60.

### 3.4 Frontend

Next.js 14 (App Router) + SWR. Pages:

- `/` **Arrival simulator** — estimate card with the green/amber/red band and evidence.
- `/register` **New patient entry** — triage priority, predicted pathway, wait estimate.
- `/stations` **Station overview** — per-station historical waits and weekly pattern.
- `/agent` **AI Agent Console** — runs the supervisor multi-agent analysis and shows each step.
- `/qa` **SOP Q&A** — staff chat over ingested SOP documents (shadcn/ui + Tailwind).
- `/documents` **Knowledge Base** — upload, re-ingest, merge and remove SOP documents.

## 4. Non-functional Requirements

- **NFR-1** Ingestion of the full 237K rows ≤ 15 min (batched embedding calls). *Not
  re-measured since embeddings moved to the OpenAI API.*
- **NFR-2** `/arrivals/estimate` P95 latency ≤ 3 s (excluding LLM cold start).
- **NFR-3** No PHI: datasets are already anonymized; never commit raw or derived data
  (`data/`, `Clean_Dataset/`, `*.xlsx` are gitignored).
- **NFR-4** *(revised)* `OPENAI_API_KEY` is required: retrieval needs it for embeddings, and
  the SOP Q&A agent has no fallback. Only the final LLM *phrasing* step is optional — the
  wait-estimate and supervisor graphs return template text if that call fails. v1.0
  required the statistics path to work with no key; that no longer holds.
- **NFR-5** Idempotent ingestion (safe to re-run).

## 5. Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Excel → Qdrant ingestion + verified nearest-event search | done |
| 2 | LangGraph agent + FastAPI estimate endpoint | done |
| 3 | Next.js dashboard | done (six pages, §3.4) |
| 4 | Queue-depth integration, nurse override (LangGraph interrupt) | partly: queue depth comes from the replayed dataset (§6.1), there is no live feed; nurse override not built |
| 5 | Evaluation on hold-out data | partly: wait-model audit and LSTM back-test done (§7); RAG estimate not yet evaluated |

## 6. Trained models and multi-agent architecture

### 6.1 Digital twin and trained models

There is no live feed, so `backend/app/services/livestate.py` reconstructs clinic state
"as of" any timestamp from `Clean_Dataset/cleaned_events.csv`: queue depth (patients whose
wait has started but whose service has not), recent completed waits, and 15-minute bucket
features. A timestamp outside the dataset's range (2024-10-01 → 2025-09-30) is mapped to
the nearest date with the same weekday and time of day, and the result is labelled as a
pattern replay.

Models (`Models/`, loaded by `models_service.py`):

- **Wait time — LightGBM**, one global model with the service category as an input
  (`Models/3_Trained_Models/deployment_global_model/`): a point model on log1p(wait
  minutes), P10/P50/P90 quantile models with a calibration shift, and a breach classifier
  (limits: visual acuity 20 min, consultation 45, visual field test 60, others 30).
  It replaced the earlier XGBoost model, whose file is kept only for
  `scripts/audit_wait_model.py`.
- **Congestion — Gaussian HMMs**, one per service category (9), three states
  (Low/Medium/High), inferred from recent waits (`Models/clinic_hmms.pkl`).
- **Queue forecast — LSTM**, 2 layers, 8 × 15-minute buckets in, 4 buckets (one hour) out
  (`Models/lstm_queue_forecast.pt`).

### 6.2 Supervisor multi-agent graph

Implemented in `backend/app/agents/supervisor_graph.py`, following
`MediQ_AI_Agents_Architecture_Fig2.png`. One supervisor acts in four phases:

- **supervisor** — reads the objective and delegates, in parallel, only to the specialists
  it calls for. If it cannot tell what is being asked it delegates to nobody and returns a
  clarifying question.
- **Waiting Time Analysis Agent** — LightGBM estimate + Qdrant RAG evidence; consensus of both.
- **Bottleneck Detection Agent** — HMM congestion state per category.
- **Queue Depth Forecasting Agent** — LSTM forecast, 4 × 15-minute buckets ahead.
- **Resource Planning Agent** — allocation options from live load shares.
- **supervisor_review** — checks who reported and decides whether the predictive agent can
  run (it needs the waiting-time and queue-forecast findings).
- **Predictive Analytics Agent** — calls no model; turns the specialists' findings into a
  trend-adjusted wait projection, an uncertainty band, a near-term outlook and a risk score.
- **supervisor_integrate** — fuses the findings.
- **supervisor_validate** — LightGBM-vs-RAG agreement, HMM/forecast consistency, numeric
  confidence score.
- **decision_support → reporting** — prioritized recommendations and the report. All
  numbers are computed in code; the LLM only phrases the prose.

API: `POST /api/v1/agent/analyze`, and `/agent/analyze/stream` (Server-Sent Events, one
event per step) · UI: `/agent`.

### 6.3 SOP & Healthcare Q&A agent

`backend/app/agents/qa_graph.py` — a tool-calling agent (`retrieve → agent ⇄ tools`) over
the `mediq_sop_docs` collection, using the same embedder and reranker. A first search
always runs; the LLM may search again with refined queries. Answers are styled by persona
(`qa_personas.py`), cite their sources, and are marked "not covered" when the best rerank
score is below `qa_covered_min_score`. The page is for staff. API: `/api/v1/qa/*` and
`/api/v1/documents/*` · UI: `/qa`, `/documents`.

### 6.4 Registration, triage and pathway

`POST /api/v1/patients/register` runs rule-based triage (P1–P3), predicts the visit
pathway, and returns the LightGBM estimate as the headline with the RAG estimate as
supporting evidence. See `PATHWAY_TRIAGE.md`.

## 7. Evaluation

**Done** (scripts in `backend/scripts/`, results in `backend/scripts/backtest_results/`):

- **Wait model** (`audit_lgbm_global.py`, summary in `Models/lgbm_app_profile.json`). The
  models were trained on data through 2025-05-30, validated to 2025-07-30, and tested on
  2025-07-31 → 2025-09-30. On the hold-out the LightGBM model scores MAE 11.4 min (RMSE
  19.0, R² 0.51) with all its training inputs. The app cannot supply some of them (room,
  visit history), so as the app calls it the MAE is 14.8 min with an appointment time and
  15.6 min without. The app shows the figure that applies as the "typical error".
- **Queue forecast** (`backtest_early_warning.py`, `backtest_early_warning_retrained.py`).
  The LSTM's own metadata reports validation MAE 2.11 against a baseline of 4.29; the
  back-tests examine how it performs inside the app's pipeline and as an early warning.

**Not done yet:**

- RAG estimate: hold out the final month of the raw export (2026-04-04 → 2026-05-04), build
  the Qdrant index from earlier data only, and report MAE and P90 absolute error in minutes
  against a global median and a per-station rolling median.
- Resolve the date mismatch between the two datasets (§2.2) so the model and RAG estimates
  can be evaluated on the same period.
- Pathway accuracy and total-visit-time error (`PATHWAY_TRIAGE.md` §7).
