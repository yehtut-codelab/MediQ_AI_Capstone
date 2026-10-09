"""SOP & Healthcare Q&A agent — a genuine tool-calling LLM agent (LangGraph ReAct
pattern), unlike WaitEstimateGraph/supervisor_graph where the LLM only rephrases
numbers computed upstream. Here the LLM decides whether/what to search and
reasons over retrieved SOP passages; there is no template fallback, since
open-ended question answering can't be computed deterministically.

retrieve -> agent -(tool_calls?)-> tools -> agent -(no tool_calls)-> END
"""

from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.qa_personas import PERSONAS, build_system_prompt, resolve_style
from app.config import settings
from app.services.embedder import embed_one
from app.services.qdrant_service import get_client
from app.services.reranker import rerank_chunks
from app.services.sop_service import search_sop


class QAState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    citations: Annotated[list[dict], lambda a, b: a + b]
    # Resolved persona / answer style (see qa_personas.py); set once per request.
    persona: str
    length: str
    register: str


INITIAL_SEARCH_ID = "call_initial_search"


def _search(query: str) -> tuple[str, list[dict]]:
    """Run one SOP search; returns (tool-message text, citations)."""
    hits = search_sop(get_client(), embed_one(query), limit=settings.sop_retrieval_k)
    top = rerank_chunks(query, hits, top_n=settings.sop_rerank_top_n)

    if not top:
        content = "No relevant SOP passages found for this query."
        citations: list[dict] = []
    else:
        content = "\n\n---\n\n".join(
            f"[{i + 1}] {h['document_name']}\n{h['text']}" for i, h in enumerate(top)
        )
        citations = [
            {
                "title": h["document_name"],
                "snippet": h["text"][:280],
                "score": round(h["rerank_score"], 4),
            }
            for h in top
        ]
    return content, citations


@tool
def search_sop_docs(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Search the SOP / healthcare-procedure knowledge base for passages relevant to `query`."""
    content, citations = _search(query)
    return Command(update={
        "citations": citations,
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
    })


def initial_retrieve(state: QAState) -> QAState:
    """Search with the user's question before the first LLM call, saving the
    round-trip where the model would only decide to search. Recorded as a normal
    tool-call pair so the model treats it as its own search and can refine."""
    question = state["messages"][-1].content
    content, citations = _search(question)
    return {
        "citations": citations,
        "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "search_sop_docs", "args": {"query": question}, "id": INITIAL_SEARCH_ID,
            }]),
            ToolMessage(content=content, tool_call_id=INITIAL_SEARCH_ID),
        ],
    }


@lru_cache(maxsize=1)
def get_llm():
    # Responses API: reasoning models (e.g. gpt-6-luna) reject function tools on
    # /v1/chat/completions when reasoning is enabled.
    # max_tokens covers reasoning + answer, so it is sized above the old 500.
    return ChatOpenAI(model=settings.llm_model, max_tokens=2000, use_responses_api=True,
                      reasoning={"effort": settings.qa_reasoning_effort, "summary": "auto"},
                      api_key=settings.openai_api_key).bind_tools([search_sop_docs])


def call_model(state: QAState) -> QAState:
    persona, length, register = resolve_style(
        state.get("persona"), state.get("length"), state.get("register"))
    system = build_system_prompt(persona, length, register)
    response = get_llm().invoke([SystemMessage(system), *state["messages"]])
    return {"messages": [response]}


def build_qa_graph():
    g = StateGraph(QAState)
    g.add_node("retrieve", initial_retrieve)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode([search_sop_docs]))
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


qa_graph = build_qa_graph()


FOLLOWUP_PROMPT = (
    "You suggest follow-up questions for an eye-clinic SOP assistant. Given the user's "
    "question, the assistant's answer and the knowledge-base passages it retrieved, write "
    "3 short follow-up questions the user is likely to ask next. Each must be answerable "
    "from the passages, must not repeat what the answer already covers, and must be "
    "self-contained (name the procedure or topic — no 'it' or 'that'), because each "
    "question is sent to the assistant on its own without the earlier conversation. "
    "Phrase them the way the stated user would ask, in Singapore/British English."
)


class FollowUps(BaseModel):
    questions: list[str] = Field(description="Exactly 3 follow-up questions")


@lru_cache(maxsize=1)
def get_followup_llm():
    # Separate, tool-free call with no reasoning so it adds little after the answer.
    return ChatOpenAI(model=settings.llm_model, max_tokens=400, use_responses_api=True,
                      reasoning={"effort": "none"},
                      api_key=settings.openai_api_key).with_structured_output(FollowUps)


def suggest_followups(question: str, answer: str, passages: list[str],
                      persona_id: str | None = None) -> list[str]:
    """Up to 3 follow-up questions grounded in the retrieved passages; [] on failure,
    since suggestions are optional and must never break the answer."""
    if not passages:
        return []
    context = "\n\n---\n\n".join(passages)
    user = PERSONAS.get(persona_id or "", None)
    who = user.audience if user else "clinic staff"
    try:
        result = get_followup_llm().invoke([
            SystemMessage(FOLLOWUP_PROMPT),
            HumanMessage(f"User: {who}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
                         f"\n\nPassages:\n{context}"),
        ])
    except Exception:
        return []
    return [q.strip() for q in result.questions if q.strip()][:3]
