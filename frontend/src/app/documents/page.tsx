"use client";

import { useEffect, useRef, useState } from "react";
import type { Topic } from "../qa/types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type DocStatus = "uploaded" | "processing" | "ingested" | "failed";

type DocumentOut = {
  id: string;
  name: string;
  ext: string;
  size_bytes: number;
  status: DocStatus;
  chunk_count: number;
  error_message: string | null;
  uploaded_at: string;
  ingested_at: string | null;
};

const STATUS_LABEL: Record<Exclude<DocStatus, "ingested">, string> = {
  failed: "Failed",
  processing: "Stuck processing",
  uploaded: "Not indexed",
};

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}

async function apiError(res: Response): Promise<Error> {
  const detail = await res.json().catch(() => null);
  return new Error(detail?.detail ?? `API ${res.status}`);
}

export default function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [topics, setTopics] = useState<Topic[] | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  // id of the document or topic an action is running on
  const [busyId, setBusyId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      const [docsRes, topicsRes] = await Promise.all([
        fetch(`${API}/api/v1/documents`),
        fetch(`${API}/api/v1/qa/topics`),
      ]);
      if (!docsRes.ok) throw await apiError(docsRes);
      if (!topicsRes.ok) throw await apiError(topicsRes);
      setDocuments(await docsRes.json());
      setTopics((await topicsRes.json()).topics);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  // Wraps an action: busy state, error display, then a refresh.
  async function run(id: string, action: () => Promise<void>) {
    setBusyId(id);
    setError("");
    try {
      await action();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  }

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file || uploading) return;
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/api/v1/documents/upload`, { method: "POST", body: form });
      if (!res.ok) throw await apiError(res);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (fileRef.current) fileRef.current.value = "";
      setUploading(false);
    }
  }

  const post = (url: string) => async () => {
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) throw await apiError(res);
  };

  function handleDelete(id: string, name: string) {
    if (!confirm(`Remove "${name}" from the knowledge base? This deletes its passages from Qdrant too.`)) return;
    run(id, async () => {
      const res = await fetch(`${API}/api/v1/documents/${id}`, { method: "DELETE" });
      if (!res.ok && res.status !== 204) throw await apiError(res);
    });
  }

  function handleMerge(t: Topic) {
    const copies = t.duplicate_count;
    if (!confirm(`Keep one copy of "${t.name}" and remove the other ${copies} identical ${copies > 1 ? "copies" : "copy"}?`)) return;
    run(t.id, post(`${API}/api/v1/documents/topics/${t.id}/merge`));
  }

  const attention = (documents ?? []).filter((d) => d.status !== "ingested");
  const duplicateCopies = (topics ?? []).reduce((n, t) => n + t.duplicate_count, 0);
  const loaded = documents !== null && topics !== null;

  return (
    <div className="kb-page">
      <div className="kb-head">
        <div>
          <h1 className="kb-title">Knowledge Base</h1>
          <p className="kb-sub">
            What the <a href="/qa">SOP assistant</a> can answer. Topics are generated from each document when it&apos;s
            uploaded.
          </p>
        </div>
        <label className={`kb-upload-btn${uploading ? " busy" : ""}`}>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.txt"
            disabled={uploading}
            onChange={handleUpload}
            className="sr-only"
          />
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 16V4" />
            <path d="M6 10l6-6 6 6" />
            <path d="M4 20h16" />
          </svg>
          {uploading ? "Uploading & indexing…" : "Upload document"}
        </label>
      </div>
      <p className="kb-hint">PDF, DOCX or TXT. Each file is indexed straight away and gets a topic card.</p>

      {error && <p className="kb-error" role="alert">{error}</p>}
      {!loaded && !error && <p className="muted">Loading knowledge base…</p>}

      {loaded && (
        <>
          <div className="kb-stats">
            <div className="kb-stat"><strong>{topics!.length}</strong><span>Topics</span></div>
            <div className="kb-stat"><strong>{documents!.length}</strong><span>Uploaded files</span></div>
            <div className={`kb-stat${duplicateCopies ? " warn" : ""}`}>
              <strong>{duplicateCopies}</strong><span>Duplicate copies to merge</span>
            </div>
            <div className={`kb-stat${attention.length ? " bad" : ""}`}>
              <strong>{attention.length}</strong><span>Failed or not searchable</span>
            </div>
          </div>

          {topics!.length === 0 && (
            <p className="muted">
              No topics yet — upload a document above. Topics for existing documents are generated in the
              background when the backend starts.
            </p>
          )}

          {topics!.map((t) => (
            <section key={t.id} className="kb-topic">
              <div className="kb-topic-head">
                <div className="kb-topic-title">
                  <div className="kb-topic-name-row">
                    <h2>{t.name}</h2>
                    <span className="qa-topic-kind">{t.kind}</span>
                  </div>
                  <p>{t.summary}</p>
                </div>
                <button
                  type="button"
                  className="kb-btn"
                  disabled={busyId === t.id}
                  onClick={() => run(t.id, post(`${API}/api/v1/documents/topics/${t.id}/regenerate`))}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M20 12a8 8 0 1 1-2.3-5.7" />
                    <path d="M20 4v5h-5" />
                  </svg>
                  {busyId === t.id ? "Working…" : "Regenerate topic"}
                </button>
              </div>

              <div className="kb-questions">
                <span className="qa-field-label">Example questions</span>
                <div className="kb-question-list">
                  {t.example_questions.map((q) => (
                    <span key={q} className="kb-question">
                      {q}
                    </span>
                  ))}
                </div>
              </div>

              {t.duplicate_count > 0 && (
                <div className="kb-dup">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
                    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect x="8" y="8" width="12" height="12" rx="2" />
                    <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
                  </svg>
                  <div>
                    <strong>{t.documents.length} files with identical content.</strong> Answers cite the same passage
                    several times.
                  </div>
                  <button type="button" disabled={busyId === t.id} onClick={() => handleMerge(t)}>
                    Keep 1, remove {t.duplicate_count} {t.duplicate_count > 1 ? "copies" : "copy"}
                  </button>
                </div>
              )}

              <div className="kb-meta">
                <span>Source: {t.source_name}</span>
                <span>{t.chunk_count} passage{t.chunk_count === 1 ? "" : "s"}</span>
                <span>Updated {formatDate(t.updated_at)}</span>
              </div>

              <details className="kb-files">
                <summary>Files ({t.documents.length})</summary>
                {t.documents.map((d) => (
                  <div key={d.id} className="kb-file-row">
                    <span className="kb-file-name">{d.name}</span>
                    <span className="kb-file-date">Uploaded {formatDate(d.uploaded_at)}</span>
                    <button
                      type="button"
                      className="kb-link danger"
                      disabled={busyId === d.id}
                      onClick={() => handleDelete(d.id, d.name)}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </details>
            </section>
          ))}

          {attention.length > 0 && (
            <section className="kb-attention">
              <span className="qa-field-label">Needs attention · not searchable</span>
              {attention.map((d) => (
                <div key={d.id} className="kb-attention-row">
                  <div className="kb-attention-name">
                    <span>{d.name}</span>
                    {d.error_message && <small>{d.error_message}</small>}
                  </div>
                  <span className={`kb-status ${d.status}`}>{STATUS_LABEL[d.status as Exclude<DocStatus, "ingested">]}</span>
                  <button
                    type="button"
                    className="kb-link"
                    disabled={busyId === d.id}
                    onClick={() => run(d.id, post(`${API}/api/v1/documents/${d.id}/reingest`))}
                  >
                    {busyId === d.id ? "Retrying…" : "Retry"}
                  </button>
                  <button
                    type="button"
                    className="kb-link danger"
                    disabled={busyId === d.id}
                    onClick={() => handleDelete(d.id, d.name)}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}
