# Agentic Framework Decision — ADR-001

**Status**: Accepted · **Date**: 2026-07-08 · **Implementation status added**: 2026-10-11

The decision below is unchanged. The *Implementation status* section at the end records
where the built system differs from what this ADR anticipated.

## Decision

**LangGraph** (on the LangChain ecosystem) is the agentic framework for this project.

## Context

The system must: retrieve nearest-neighbour historical visit events from Qdrant when a new
patient arrives, reason over them (queue depth, station sequence, time-of-day), and produce
a wait time + queue estimate with an explanation. This is a **deterministic, stateful
workflow with LLM reasoning steps** — not an open-ended multi-agent conversation.

## Options Considered

| Criterion | LangGraph | CrewAI | AutoGen | LlamaIndex Workflows |
|---|---|---|---|---|
| Deterministic graph control-flow | ✅ First-class (StateGraph) | ⚠️ Role/task abstraction, less control | ⚠️ Conversation-driven | ✅ Event-driven steps |
| Qdrant integration | ✅ `langchain-qdrant` official | Via LangChain anyway | Manual | ✅ Native |
| State management / checkpoints | ✅ Built-in persistence | ❌ Limited | ⚠️ | ⚠️ |
| Streaming into FastAPI | ✅ `astream_events` | ⚠️ | ⚠️ | ✅ |
| Human-in-the-loop (nurse override) | ✅ Built-in interrupts | ❌ | ⚠️ | ⚠️ |
| Maturity / docs / hiring signal | ✅ Industry default 2025-26 | Good | Good | Good |
| Overhead for a 4-node pipeline | Low | High (crew/role ceremony) | High | Low |

## Rationale

1. **The workflow is a graph, not a crew.** Our pipeline is: `parse patient context →
   retrieve similar events (Qdrant) → compute statistics → LLM synthesis → respond`.
   LangGraph's `StateGraph` models exactly this, with typed state and conditional edges
   (e.g., "if fewer than 20 similar events found, widen the filter and retry").
   CrewAI/AutoGen add multi-agent conversation ceremony we don't need.

2. **Qdrant is first-class.** `langchain-qdrant` gives us the vector store abstraction,
   metadata filtering, and hybrid search without custom glue.

3. **Checkpointing and interrupts** map to real clinic needs: a nurse can review/override
   an estimate before it is shown to the patient (human-in-the-loop), and conversation
   state survives across a patient's multi-station journey.

4. **Capstone defensibility**: LangGraph is the most widely adopted agent-orchestration
   framework in industry; the graded report can cite an explicit architecture rather
   than a black-box "crew".

## Consequences

- Dependencies: `langgraph`, `langchain-core`, `langchain-openai`, `qdrant-client`.
- The LLM provider is OpenAI, called through `langchain-openai` (`ChatOpenAI`); the model is
  set by `LLM_MODEL`. Embeddings use the same OpenAI account, so `OPENAI_API_KEY` is required.
- If the project later needs true multi-agent negotiation (e.g., separate scheduler +
  resource-allocation agents debating), LangGraph supports subgraphs — no migration needed.

## Implementation status (2026-10-11)

What was built on this decision, and where it differs from the text above:

- **Three graphs, all `StateGraph`s**, in `backend/app/agents/`:
  - `graph.py` — `WaitEstimateGraph`, the pipeline this ADR was written for. It gained a
    cross-encoder `rerank` node, so it is now
    `parse_context → retrieve → check_evidence →(widen, up to 2×)→ rerank → compute_stats → synthesize`.
  - `supervisor_graph.py` — a supervisor with four specialist agents and a predictive
    agent, for clinic-wide analysis. It is one graph with parallel fan-out and conditional
    edges, not subgraphs.
  - `qa_graph.py` — the SOP Q&A agent, a tool-calling loop (`retrieve → agent ⇄ tools`)
    built with LangGraph's prebuilt `ToolNode`.
- **LLM provider changed.** The original default was Claude via `langchain-anthropic`, with
  a local Ollama option, behind `init_chat_model`. The project moved to OpenAI for both the
  LLM and the embeddings, and calls `ChatOpenAI` directly. There is no offline mode.
- **Qdrant is called through `qdrant-client`, not `langchain-qdrant`.** Rationale 2 expected
  the LangChain vector-store abstraction; the code builds its own payload filters and
  queries instead. `langchain-qdrant` is still listed in `requirements.txt` but is unused.
- **Streaming is used.** The agent console and the Q&A page stream graph progress to the
  browser as Server-Sent Events, using `graph.stream(...)` rather than `astream_events`.
- **Checkpointing and interrupts are not used yet.** Rationale 3 (nurse override,
  state across a multi-station journey) is still a future item; no graph is compiled with a
  checkpointer.
