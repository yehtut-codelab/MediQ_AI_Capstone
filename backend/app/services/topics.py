"""Knowledge-base topics: what the SOP assistant can answer, per document.

Each ingested document gets a content fingerprint; identical uploads share it
and form one *group*, so duplicates collapse into a single topic. Each group
gets one generated topic card (name, kind, summary, example questions), made
once from the document text and stored on every document in the group.

`ensure_topics()` fills in whatever is missing (fingerprints, topic cards) and
is safe to call repeatedly — after an upload/reingest, and once at startup to
backfill documents ingested before topics existed.
"""

import hashlib
import json
import logging
import re
import threading
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.services import document_registry
from app.services.qdrant_service import get_client
from app.services.sop_service import document_text

log = logging.getLogger(__name__)

# One generation pass at a time: the startup backfill and an upload may overlap.
_lock = threading.Lock()

# Upload filenames that are just a UUID (bulk-ingested copies) make poor labels.
_UUID_NAME = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\.\w+)?$", re.I)

# Enough of the document for a faithful summary without a large prompt.
_MAX_CHARS = 12_000


class TopicCard(BaseModel):
    name: str = Field(description="Short topic name, 2-6 words, e.g. 'Pupil dilation before OCT'")
    kind: Literal["SOP", "Patient leaflet", "Guideline", "Policy", "Form", "Other"] = Field(
        description="What sort of document this is")
    summary: str = Field(description="One sentence, max 30 words: what questions it can answer")
    example_questions: list[str] = Field(
        description="3-5 questions clinic staff might ask that this document fully answers")


TOPIC_PROMPT = (
    "You catalogue documents in an eye clinic's SOP knowledge base so staff know what the "
    "SOP assistant can answer. From the document text, write a topic card. The summary "
    "and every example question must be fully answerable from this text alone — do not "
    "mention anything the text doesn't cover. Example questions are asked by clinic staff "
    "(nurses, technicians, doctors), never by patients — even when the document is a "
    "patient leaflet written to 'you'. Phrase them about patients in the third person "
    "('How many hours a day can patients wear contact lenses?'), never in the first "
    "person ('I', 'my', 'me'). They are self-contained (name the procedure) and use "
    "Singapore/British English."
)

# Patient-voice questions ("should I…", "my lenses") slip through on patient leaflets.
_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|me|my|mine|myself)\b")


def content_hash(text: str) -> str:
    """Fingerprint of the text, ignoring case and whitespace differences."""
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def group_id(content_hash_: str) -> str:
    return content_hash_[:16]


def _llm():
    return ChatOpenAI(model=settings.llm_model, max_tokens=600, use_responses_api=True,
                      reasoning={"effort": "none"},
                      api_key=settings.openai_api_key).with_structured_output(TopicCard)


def generate_topic(text: str) -> TopicCard:
    messages = [SystemMessage(TOPIC_PROMPT), HumanMessage(text[:_MAX_CHARS])]
    card = _llm().invoke(messages)
    patient_voice = [q for q in card.example_questions if _FIRST_PERSON.search(q)]
    if patient_voice:
        # One corrective retry, then drop any question still in the patient's voice.
        messages.append(HumanMessage(
            "These example questions are in a patient's first-person voice; rewrite every "
            "question as clinic staff asking about patients: " + " | ".join(patient_voice)))
        card = _llm().invoke(messages)
        card.example_questions = [q for q in card.example_questions if not _FIRST_PERSON.search(q)]
    return card


NOT_INDEXED = "Not searchable: no passages for this document in the knowledge base. Retry to re-index it."


def _read_text(row) -> str | None:
    """The document's indexed text from its Qdrant chunks — the same text the
    assistant answers from, and available even if the source file is gone.
    "" means the document has no chunks; None means Qdrant couldn't be read."""
    try:
        client = get_client()
        if not client.collection_exists(settings.qdrant_sop_collection):
            return None
        return document_text(client, row["id"]).strip()
    except Exception:
        log.warning("Could not read chunks of %s for topics", row["name"], exc_info=True)
        return None


def ensure_topics(regenerate_group: str | None = None) -> None:
    """Fill in missing fingerprints and topic cards for ingested documents.

    `regenerate_group` forces a fresh topic card for that one group.
    """
    if not settings.openai_api_key:
        return
    with _lock:
        rows = [r for r in document_registry.list_all() if r["status"] == "ingested"]

        # 1. Fingerprints (reads each document's chunks once, only when missing).
        texts: dict[str, str] = {}
        for r in rows:
            if not r["content_hash"]:
                text = _read_text(r)
                if text:
                    texts[r["id"]] = text
                    document_registry.set_content_hash(r["id"], content_hash(text))
                elif text == "":
                    # Marked "ingested" but nothing is searchable — say so, so the
                    # Knowledge Base page offers a retry instead of hiding it.
                    document_registry.update_status(r["id"], "failed", error_message=NOT_INDEXED)
        rows = [r for r in document_registry.list_all() if r["status"] == "ingested" and r["content_hash"]]

        # 2. One topic card per group; reuse one any member already has.
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r["content_hash"], []).append(r)
        for chash, members in groups.items():
            forced = regenerate_group == group_id(chash)
            existing = next((m["topic_json"] for m in members if m["topic_json"]), None)
            if existing and not forced:
                missing = [m["id"] for m in members if not m["topic_json"]]
                if missing:
                    document_registry.set_topic(missing, existing)
                continue
            source = members[0]
            text = texts.get(source["id"]) or _read_text(source)
            if not text:
                continue
            try:
                card = generate_topic(text)
            except Exception:
                log.warning("Topic generation failed for %s", source["name"], exc_info=True)
                continue
            document_registry.set_topic([m["id"] for m in members], card.model_dump_json())


def ensure_topics_in_background() -> None:
    threading.Thread(target=ensure_topics, name="topic-backfill", daemon=True).start()


def _display_name(rows) -> str:
    """Prefer a human filename over a UUID-named copy."""
    named = [r["name"] for r in rows if not _UUID_NAME.match(r["name"])]
    return (named or [rows[0]["name"]])[0]


def catalogue() -> list[dict]:
    """Topic groups for the UI, alphabetical. Only ingested documents with a card."""
    groups: dict[str, list] = {}
    for r in document_registry.list_all():
        if r["status"] == "ingested" and r["content_hash"] and r["topic_json"]:
            groups.setdefault(r["content_hash"], []).append(r)

    topics = []
    for chash, members in groups.items():
        card = json.loads(members[0]["topic_json"])
        # Oldest upload first: that is the copy a merge keeps.
        members = sorted(members, key=lambda r: r["uploaded_at"])
        topics.append({
            "id": group_id(chash),
            "name": card["name"],
            "kind": card["kind"],
            "summary": card["summary"],
            "example_questions": card["example_questions"][:5],
            "source_name": _display_name(members),
            "chunk_count": members[0]["chunk_count"] or 0,
            "documents": [{"id": m["id"], "name": m["name"], "uploaded_at": m["uploaded_at"]}
                          for m in members],
            "duplicate_count": len(members) - 1,
            "updated_at": max(m["topic_at"] or m["ingested_at"] or "" for m in members),
        })
    return sorted(topics, key=lambda t: t["name"].lower())


def keep_document(topic_id: str) -> tuple[str, list[str]] | None:
    """For a merge: (document to keep, documents to remove) in a topic group."""
    for t in catalogue():
        if t["id"] == topic_id:
            docs = t["documents"]
            # Keep the human-named copy if there is one, else the oldest.
            keep = next((d for d in docs if not _UUID_NAME.match(d["name"])), docs[0])
            return keep["id"], [d["id"] for d in docs if d["id"] != keep["id"]]
    return None
