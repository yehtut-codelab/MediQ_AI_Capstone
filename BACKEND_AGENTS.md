# MediQ AI — Backend Agents

Summary of the agent graphs in `backend/app/agents/` (as of 2026-10-10, commit `31dc641`).

The backend has three LangGraph graphs. Only one of them, the SOP Q&A agent, is an
LLM-driven agent; in the other two every number is computed in code and the LLM only
rewords the final text.

## The three graphs

| Graph | File | Endpoint | What it does |
|---|---|---|---|
| Supervisor multi-agent graph | `supervisor_graph.py` | `POST /api/v1/agent/analyze` | Full clinic analysis: wait estimate, bottlenecks, queue forecast, staffing options, recommendations |
| Wait Estimate graph | `graph.py` | `POST /api/v1/arrivals/estimate` | RAG estimate of one patient's wait from similar historical cases |
| SOP Q&A agent | `qa_graph.py` | `POST /api/v1/qa/ask`, `/qa/ask/stream` | Answers staff questions from the SOP knowledge base, with citations |

## All agents

### Supervisor graph (10 nodes)

| # | Agent / module | Uses | Output |
|---|---|---|---|
| 0 | Supervisor | Typical visit profile | Fills in shared context, fans out to agents 1–4 |
| 1 | Waiting Time Agent | XGBoost + Qdrant RAG | Model wait estimate, RAG median of similar visits, current queue depth |
| 2 | Bottleneck Detection Agent | Per-category HMM | Congestion state (Low/Medium/High) per station; flags bottlenecks |
| 3 | Queue Depth Forecast Agent | LSTM | Next-hour queue depth (4 × 15 min) and trend (rising/falling/stable) |
| 4 | Resource Planning Agent | Rules on queue load share | Up to 4 options (open a room, reassign floating staff) |
| 5 | Predictive Analytics Agent | Findings of agents 1–4, no model of its own | Trend-adjusted wait, uncertainty range, outlook, risk score 0–1 |
| 6 | Result Fusion module | — | Consensus wait (mean of XGBoost and RAG), spread between them |
| 7 | Validation & Critique Agent | Consistency checks | Confidence score and label (XGBoost vs RAG agreement, HMM vs forecast) |
| 8 | Decision Support module | Rules | Up to 6 prioritised recommendations |
| 9 | Reporting | Optional LLM rewrite | Summary for a nurse manager, alerts, full findings |

Agents 1–4 run in parallel; agent 5 waits for all four.

Details worth knowing:

- **Bottleneck rule:** a station is flagged if its HMM state is High, or Medium with a queue depth of 10 or more.
- **Trend rule:** rising if the first forecast bucket is more than 15% above the current depth, falling if more than 15% below.
- **Predicted wait:** XGBoost estimate × 1.15 (rising), × 0.85 (falling), × 1.0 (stable / no data).
- **Uncertainty:** root-sum-square of the XGBoost and LSTM validation MAEs.
- **Risk score:** 0.4 × rising trend + 0.4 × flagged stations (capped at 3) + 0.2 × resource options (capped at 4); high ≥ 0.66, medium ≥ 0.33.
- **Confidence:** starts at 1.0; −0.3 if XGBoost and RAG differ by more than 50% of the consensus, −0.15 if there is no RAG evidence, −0.2 if HMM and forecast contradict, −0.15 if there is no forecast data; floor 0.2.

### Wait Estimate graph (6 nodes, a retrieval pipeline rather than an agent)

`parse_context` → `retrieve` → `widen` (up to 2 times if fewer than 20 hits) → `rerank`
(cross-encoder, keeps top 10 of 50) → `compute_stats` (median, P10/P75/P90, confidence) →
`synthesize` (optional LLM rewrite).

Widening ladder: 0 = exact weekday + hour ± 2; 1 = weekday/weekend class + hour ± 2;
2 = no time filter.

### SOP Q&A agent (3 nodes, ReAct tool-calling loop)

- `retrieve` — searches the SOP collection with the user's question before the first LLM call.
- `agent` — the LLM (`gpt-6-luna`, Responses API, with reasoning), using the persona system
  prompt from `qa_personas.py`. It decides whether to search again or answer.
- `tools` — the one tool, `search_sop_docs` (Qdrant search + rerank, returns citations).
- Outside the graph, `suggest_followups` makes a separate LLM call for 3 follow-up questions.

The five personas (clinical staff, clinician, patient explainer, front desk, trainee) are
answer styles for this one agent, not separate agents.

## Architecture diagram

```
                              Next.js frontend
        /agent console          / arrival simulator            /qa
              │                        │                        │
══════════════╪════════════════════════╪════════════════════════╪═══════════════
 FastAPI      │ /agent/analyze         │ /arrivals/estimate     │ /qa/ask(/stream)
              ▼                        ▼                        ▼
┌──────────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ SUPERVISOR GRAPH         │ │ WAIT ESTIMATE GRAPH  │ │ SOP Q&A AGENT        │
│                          │ │                      │ │                      │
│       supervisor         │ │    parse_context     │ │      retrieve        │
│           │ fan-out      │ │          │           │ │         │            │
│  ┌──────┬─┴────┬──────┐  │ │          ▼           │ │         ▼            │
│  ▼      ▼      ▼      ▼  │ │      retrieve ◄──┐   │ │       agent ◄───┐    │
│ wait  bottle  queue  res.│ │          │       │   │ │    (LLM+persona)│    │
│ time  neck    fcst   plan│ │   < 20 hits?   widen │ │         │       │    │
│ (1)   (2)     (3)    (4) │ │          │─yes───┘   │ │   tool call?    │    │
│  └──────┴──┬───┴──────┘  │ │          │ no (max 2)│ │     │yes  │no   │    │
│            ▼ fan-in      │ │          ▼           │ │     ▼     ▼     │    │
│   predictive agent (5)   │ │       rerank         │ │   tools   END   │    │
│            ▼             │ │          ▼           │ │ search_sop_docs─┘    │
│     result fusion        │ │    compute_stats     │ │                      │
│            ▼             │ │          ▼           │ │ + suggest_followups  │
│  validation & critique   │ │     synthesize       │ │   (separate LLM call)│
│            ▼             │ │   (optional LLM)     │ │                      │
│     decision support     │ │          ▼           │ │                      │
│            ▼             │ │         END          │ │                      │
│  reporting (optional LLM)│ │                      │ │                      │
│            ▼             │ │                      │ │                      │
│           END            │ │                      │ │                      │
└─────┬──────────────┬─────┘ └──────────┬───────────┘ └──────────┬───────────┘
      │              │                  │                        │
══════╪══════════════╪══════════════════╪════════════════════════╪══════════════
 Services / data     │                  │                        │
      ▼              ▼                  ▼                        ▼
┌───────────────┐ ┌─────────────────────────────────┐ ┌──────────────────────┐
│ Trained models│ │ Qdrant: ttsh_wait_events        │ │ Qdrant: mediq_sop_docs│
│ XGBoost  (1)  │ │ 5,000 historical wait events    │ │ SOP document chunks  │
│ HMM      (2)  │ │ used by agent (1) and the       │ │                      │
│ LSTM     (3)  │ │ Wait Estimate graph             │ │                      │
│ + livestate   │ └─────────────────────────────────┘ └──────────────────────┘
│ (Clean_Dataset│
│  replay)      │   Shared: OpenAI embeddings (text-embedding-3-small),
└───────────────┘   local cross-encoder reranker, OpenAI LLM (gpt-6-luna)
```

The numbers (1)–(5) in the supervisor graph match the agent table above. The arrival
simulator page also calls `/queue/simulate`, which runs the same XGBoost, HMM and LSTM
models directly without going through a graph.
