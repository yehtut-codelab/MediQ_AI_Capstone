"use client";

import { useEffect, useMemo, useRef, useState, type ComponentProps, type CSSProperties, type ReactNode } from "react";
import {
  ArrowDown, ArrowUp, Check, Copy, CornerDownRight, PanelLeftClose, PanelLeftOpen, Pencil, Plus, RefreshCw, Search, Square,
  Trash2, X,
} from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import AnswerFeedback from "./AnswerFeedback";
import Markdown from "./Markdown";
import { pickPrompts, themeFor } from "./personaTheme";
import SetupHero from "./SetupHero";
import StyleChip from "./StyleChip";
import TopicsMenu, { TopicHelp } from "./TopicsMenu";
import type { Catalogue, Message, PersonaOption, Prefs, Session, StyleOption, Topic } from "./types";
import { Orb } from "./ui";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STORAGE_KEY = "mediq_qa_sessions";
const PREFS_KEY = "mediq_qa_prefs";
const SIDEBAR_KEY = "mediq_qa_sidebar";

/** Small ghost icon button with a tooltip, used in the message action rows. */
const ACTION_CLASS = cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "size-8 text-muted-foreground");

/** Copies a message's text (answers as Markdown); shows a tick for a moment after. */
function CopyButton({ text, label = "Copy answer" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked (e.g. insecure context) — nothing useful to show */
    }
  }
  return (
    <Tooltip>
      <TooltipTrigger type="button" aria-label={copied ? "Copied" : label} onClick={copy} className={ACTION_CLASS}>
        {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
      </TooltipTrigger>
      <TooltipContent side="bottom">{copied ? "Copied" : "Copy"}</TooltipContent>
    </Tooltip>
  );
}

function EditButton({ onClick, disabled }: { onClick: () => void; disabled: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger type="button" aria-label="Edit question" onClick={onClick} disabled={disabled} className={ACTION_CLASS}>
        <Pencil aria-hidden="true" />
      </TooltipTrigger>
      <TooltipContent side="bottom">Edit</TooltipContent>
    </Tooltip>
  );
}

/** Icon-only button with a tooltip. The Radix trigger is the <button> itself —
 * wrapping a shadcn Button via `asChild` would lose the ref on React 18. */
function IconTip({
  label,
  variant = "ghost",
  className,
  children,
  ...props
}: {
  label: string;
  variant?: "ghost" | "secondary";
  className?: string;
  children: ReactNode;
} & Omit<ComponentProps<"button">, "children">) {
  return (
    <Tooltip>
      <TooltipTrigger
        type="button"
        aria-label={label}
        className={cn(buttonVariants({ variant, size: "icon" }), "size-9 text-muted-foreground [&_svg]:size-[18px]", className)}
        {...props}
      >
        {children}
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-60">{label}</TooltipContent>
    </Tooltip>
  );
}

function loadPrefs(): Prefs | null {
  try {
    const raw = window.localStorage.getItem(PREFS_KEY);
    return raw ? (JSON.parse(raw) as Prefs) : null;
  } catch {
    return null;
  }
}

function savePrefs(prefs: Prefs) {
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* preference just won't persist */
  }
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function isToday(ts: number) {
  const d = new Date(ts);
  const t = new Date();
  return d.toDateString() === t.toDateString();
}

function loadSessions(): Session[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session[]) : [];
  } catch {
    return [];
  }
}

function saveSessions(sessions: Session[]) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    /* storage full/unavailable — chat still works for this tab session */
  }
}

let nextMsgId = 1;

export default function SopQA() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  // Empty for the server render; filled on mount to avoid a hydration mismatch.
  const [prompts, setPrompts] = useState<string[]>([]);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  // Chat list collapsed to an icon rail; remembered per browser.
  const [railMode, setRailMode] = useState(false);
  // What the knowledge base covers (GET /qa/topics); empty until loaded.
  const [topics, setTopics] = useState<Topic[]>([]);
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  // The setup screen's choice, used for the next new chat.
  const [prefs, setPrefs] = useState<Prefs | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // True once history has been read from storage (before that, never save).
  const hydrated = useRef(false);
  // Aborts the in-flight answer stream (Stop button).
  const abortRef = useRef<AbortController | null>(null);
  // Messages created in this tab: only these animate in, not a reopened chat's history.
  const freshIds = useRef(new Set<number>());

  useEffect(() => {
    hydrated.current = true;
    setSessions(loadSessions());
    setPrompts(pickPrompts(undefined));
    try {
      setRailMode(window.localStorage.getItem(SIDEBAR_KEY) === "collapsed");
    } catch {
      /* storage unavailable — start expanded */
    }

    fetch(`${API}/api/v1/qa/topics`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data: { topics: Topic[] }) => setTopics(data.topics))
      .catch(() => {
        /* topics are optional: the Topics button and not-covered help stay hidden */
      });

    fetch(`${API}/api/v1/qa/personas`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((cat: Catalogue) => {
        setCatalogue(cat);
        // Keep a saved choice only if the server still offers that persona/style.
        const saved = loadPrefs();
        const has = (list: StyleOption[] | PersonaOption[], id?: string) =>
          !!id && list.some((o) => o.id === id);
        const persona = cat.personas.find((p) => p.id === saved?.persona)
          ?? cat.personas.find((p) => p.id === cat.default_persona)
          ?? cat.personas[0];
        setPrefs({
          persona: persona.id,
          length: has(cat.lengths, saved?.length) ? saved!.length : persona.default_length,
          technical_level: has(cat.technical_levels, saved?.technical_level)
            ? saved!.technical_level
            : persona.default_technical_level,
        });
        setPrompts(pickPrompts(persona.id));
      })
      .catch(() => {
        /* setup screen falls back to plain suggestions; backend uses its default persona */
      });
  }, []);

  function updatePrefs(change: Partial<Prefs>) {
    if (!prefs || !catalogue) return;
    let next = { ...prefs, ...change };
    // Switching persona resets length/technical level to that persona's defaults.
    if (change.persona && change.persona !== prefs.persona) {
      const p = catalogue.personas.find((x) => x.id === change.persona);
      if (p) next = { persona: p.id, length: p.default_length, technical_level: p.default_technical_level };
      setPrompts(pickPrompts(next.persona));
    }
    setPrefs(next);
    savePrefs(next);
  }

  const activeSession = sessions.find((s) => s.id === activeId) ?? null;
  const messages = activeSession?.messages ?? [];
  const inSetup = messages.length === 0;

  // Settings in force: a running chat's own snapshot, else the setup choice.
  // Chats saved before personas existed fall back to the default persona.
  const effective: Prefs | null = useMemo(() => {
    if (!activeSession || inSetup) return prefs;
    const personaId = activeSession.persona ?? catalogue?.default_persona;
    const persona = catalogue?.personas.find((p) => p.id === personaId);
    if (!personaId) return null;
    return {
      persona: personaId,
      length: activeSession.length ?? persona?.default_length ?? "balanced",
      technical_level: activeSession.technical_level ?? persona?.default_technical_level ?? "standard",
    };
  }, [activeSession, inSetup, prefs, catalogue]);

  const effectivePersona = catalogue?.personas.find((p) => p.id === effective?.persona);
  const theme = themeFor(effective?.persona);

  const todaySessions = useMemo(
    () => sessions.filter((s) => isToday(s.updatedAt)).sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions]
  );
  const olderCount = sessions.length - todaySessions.length;

  const searchResults = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return [];
    return sessions
      .filter(
        (s) =>
          s.title.toLowerCase().includes(q) ||
          s.messages.some((m) => m.text.toLowerCase().includes(q))
      )
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }, [sessions, searchQuery]);

  // Persist chat history. Streaming updates arrive per token, so writes are
  // debounced while an answer streams and flushed as soon as it ends — one
  // localStorage write per answer instead of one per token.
  useEffect(() => {
    if (!hydrated.current) return; // don't overwrite storage with the initial []
    const t = setTimeout(() => saveSessions(sessions), streaming ? 800 : 0);
    return () => clearTimeout(t);
  }, [sessions, streaming]);

  // Follow new content only while the reader is at the bottom (chat-app
  // convention); if they scroll up to read, stay put and offer "jump to latest".
  const stickToBottom = useRef(true);
  const [showJump, setShowJump] = useState(false);
  // Inline edit of a sent question: which message, and the draft text.
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");

  function onThreadScroll() {
    const el = threadRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    stickToBottom.current = atBottom;
    setShowJump(!atBottom && !inSetup);
  }

  function scrollToBottom(smooth = false) {
    const el = threadRef.current;
    if (!el) return;
    stickToBottom.current = true;
    setShowJump(false);
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }

  useEffect(() => {
    if (!inSetup && stickToBottom.current) scrollToBottom();
  }, [messages, loading, inSetup]);

  // Opening a chat, or starting a new one, lands at the latest message with the composer ready.
  useEffect(() => {
    stickToBottom.current = true;
    setEditingId(null);
    requestAnimationFrame(() => {
      scrollToBottom();
      textareaRef.current?.focus();
    });
  }, [activeId]);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }

  function updateSession(id: string, fn: (s: Session) => Session) {
    setSessions((prev) => prev.map((s) => (s.id === id ? fn(s) : s)));
  }

  function newChat() {
    setActiveId(null);
    setSearchOpen(false);
    setInput("");
    setPrompts(pickPrompts(prefs?.persona));
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  // pendingDelete holds a session id, or "all"; the first click arms it, the second deletes.
  function deleteSession(id: string) {
    if (loading) return;
    setSessions((prev) => prev.filter((s) => s.id !== id));
    setPendingDelete(null);
    if (id === activeId) newChat();
  }

  function clearAllSessions() {
    if (loading) return;
    setSessions([]);
    setPendingDelete(null);
    setSearchQuery("");
    newChat();
  }

  function personaLabel(s: Session) {
    const id = s.persona ?? catalogue?.default_persona;
    return catalogue?.personas.find((p) => p.id === id)?.name;
  }

  function renderSessionRow(s: Session, when: string) {
    const armed = pendingDelete === s.id;
    const who = personaLabel(s);
    return (
      <div key={s.id} className={`qa-session-row${armed ? " armed" : ""}`}>
        <button
          className={`qa-session-item${s.id === activeId ? " active" : ""}`}
          onClick={() => openSession(s.id)}
        >
          <span className="qa-session-item-title">
            <span className="qa-session-dot" style={{ background: themeFor(s.persona).accent }} aria-hidden="true" />
            {s.title}
          </span>
          <span className="qa-session-item-time">{who ? `${who} · ${when}` : when}</span>
        </button>
        <button
          className="qa-session-delete"
          disabled={loading}
          title={armed ? "Click again to delete" : "Delete conversation"}
          aria-label={armed ? `Confirm delete "${s.title}"` : `Delete "${s.title}"`}
          onClick={() => (armed ? deleteSession(s.id) : setPendingDelete(s.id))}
          onBlur={() => armed && setPendingDelete(null)}
        >
          {armed ? "Delete?" : <X size={14} strokeWidth={2.4} aria-hidden="true" />}
        </button>
      </div>
    );
  }

  function setSidebarCollapsed(collapsed: boolean) {
    setRailMode(collapsed);
    if (collapsed) {
      setSearchOpen(false);
      setPendingDelete(null);
    }
    try {
      window.localStorage.setItem(SIDEBAR_KEY, collapsed ? "collapsed" : "expanded");
    } catch {
      /* choice just won't persist */
    }
  }

  function openSession(id: string) {
    setActiveId(id);
    setSearchOpen(false);
    setSearchQuery("");
  }

  async function ask(question: string) {
    if (!question.trim() || loading) return;

    let sessionId = activeId;
    const userMsg: Message = { id: nextMsgId++, role: "user", text: question, time: now() };
    freshIds.current.add(userMsg.id);
    stickToBottom.current = true; // sending always follows the new answer
    // Style for this question: a new chat snapshots the setup choice; a running
    // chat keeps its own persona and current length/technical level.
    const style: Prefs | null = activeSession && !inSetup ? effective : prefs;

    if (!sessionId || !activeSession) {
      sessionId = crypto.randomUUID();
      const fresh: Session = {
        id: sessionId,
        title: question.slice(0, 60),
        messages: [userMsg],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...(style ?? {}),
      };
      setSessions((prev) => [fresh, ...prev]);
      setActiveId(sessionId);
    } else {
      updateSession(sessionId, (s) => ({
        ...s,
        ...(s.messages.length === 0 && style ? style : {}),
        messages: [...s.messages, userMsg],
        updatedAt: Date.now(),
      }));
    }

    setInput("");
    setLoading(true);
    requestAnimationFrame(autoResize);

    const finish = (msg: Message) =>
      updateSession(sessionId as string, (s) => ({
        ...s,
        messages: [...s.messages, msg],
        updatedAt: Date.now(),
      }));

    // Streamed reply: one assistant message, created on the first event and
    // patched in place. Date.now() keeps its id clear of ids restored from storage.
    const replyId = Date.now();
    freshIds.current.add(replyId);
    let started = false;
    const controller = new AbortController();
    abortRef.current = controller;
    const patchReply = (fn: (m: Message) => Message) => {
      if (!started) {
        started = true;
        setStreaming(true);
        finish(fn({ id: replyId, role: "assistant", text: "", thinking: [], time: now() }));
        return;
      }
      updateSession(sessionId as string, (s) => ({
        ...s,
        messages: s.messages.map((m) => (m.id === replyId ? fn(m) : m)),
        updatedAt: Date.now(),
      }));
    };

    try {
      const res = await fetch(`${API}/api/v1/qa/ask/stream`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        // No style (catalogue failed to load) -> backend uses its default persona.
        body: JSON.stringify({ question, ...(style ?? {}) }),
      });

      if (res.status === 503) {
        finish({
          id: nextMsgId++,
          role: "assistant",
          text: "SOP Q&A is not available — the backend needs OPENAI_API_KEY configured to answer questions.",
          isError: true,
          time: now(),
        });
        return;
      }
      if (!res.ok || !res.body) {
        // Prefer the server's own reason (FastAPI puts it in `detail`) over a bare status code.
        let reason = `API ${res.status}`;
        try {
          const body = await res.json();
          const detail = Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail;
          if (typeof detail === "string" && detail) reason = detail.replace(/^Value error, /, "");
        } catch {
          /* not JSON — keep the status code */
        }
        throw new Error(reason);
      }

      // Server-Sent Events: "data: {json}\n\n" frames (see /qa/ask/stream)
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          if (!frame.startsWith("data: ")) continue;
          const ev = JSON.parse(frame.slice(6));
          if (ev.type === "start") {
            patchReply((m) => ({ ...m, personaName: ev.persona_name }));
          } else if (ev.type === "thinking") {
            patchReply((m) => ({ ...m, thinking: [...(m.thinking ?? []), ev.step] }));
          } else if (ev.type === "token") {
            patchReply((m) => ({ ...m, text: m.text + ev.text }));
          } else if (ev.type === "done") {
            patchReply((m) => ({
              ...m,
              text: ev.answer,
              sources: ev.sources,
              thinking: ev.thinking,
              covered: ev.covered,
            }));
          } else if (ev.type === "followups") {
            patchReply((m) => ({ ...m, followups: ev.questions }));
          } else if (ev.type === "error") {
            throw new Error(ev.message);
          }
        }
      }
    } catch (err) {
      if (controller.signal.aborted) {
        // Stopped by the user: keep whatever had streamed, marked as stopped.
        if (started) patchReply((m) => ({ ...m, stopped: true }));
        else finish({ id: replyId, role: "assistant", text: "", stopped: true, time: now() });
      } else {
        const reason = err instanceof Error ? err.message : String(err);
        const text = `Sorry, I couldn't answer that: ${reason}`;
        if (started) {
          patchReply((m) => ({ ...m, text, isError: true }));
        } else {
          finish({ id: replyId, role: "assistant", text, isError: true, time: now() });
        }
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
      setStreaming(false);
    }
  }

  function stopAnswer() {
    abortRef.current?.abort();
  }

  /** Thumbs up/down: kept on the message (so it survives reloads) and sent to the
   * backend for answer-quality review. Fire-and-forget: a failed POST never blocks the chat. */
  function rateAnswer(m: Message, rating: "up" | "down" | undefined, reason?: string) {
    if (!activeSession) return;
    const idx = activeSession.messages.findIndex((x) => x.id === m.id);
    const question = [...activeSession.messages.slice(0, idx)].reverse().find((x) => x.role === "user")?.text ?? "";
    const nextReason = rating === "down" ? reason ?? (m.feedback === "down" ? m.feedbackReason : undefined) : undefined;
    updateSession(activeSession.id, (s) => ({
      ...s,
      messages: s.messages.map((x) => (x.id === m.id ? { ...x, feedback: rating, feedbackReason: nextReason } : x)),
    }));
    fetch(`${API}/api/v1/qa/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: activeSession.id,
        message_id: String(m.id),
        rating: rating ?? null,
        reason: nextReason ?? null,
        question,
        answer: m.text,
        persona: activeSession.persona ?? null,
      }),
    }).catch(() => {
      /* feedback is best-effort */
    });
  }

  /** Redo the latest answer: drop it and its question, then ask the same question again. */
  function regenerate() {
    if (!activeSession || loading) return;
    const msgs = activeSession.messages;
    const userIdx = msgs.map((m) => m.role).lastIndexOf("user");
    if (userIdx < 0) return;
    const question = msgs[userIdx].text;
    updateSession(activeSession.id, (s) => ({ ...s, messages: s.messages.slice(0, userIdx) }));
    ask(question);
  }

  function startEdit(m: Message) {
    setEditingId(m.id);
    setEditText(m.text);
  }

  /** Replace a sent question: drop it and everything after, then ask the new text
   * (ChatGPT's edit-and-resend). Unchanged text just closes the editor. */
  function submitEdit(msgId: number) {
    const text = editText.trim();
    const idx = activeSession?.messages.findIndex((m) => m.id === msgId) ?? -1;
    setEditingId(null);
    if (!activeSession || idx < 0 || !text || loading || text === activeSession.messages[idx].text) return;
    updateSession(activeSession.id, (s) => ({
      ...s,
      title: idx === 0 ? text.slice(0, 60) : s.title,
      messages: s.messages.slice(0, idx),
    }));
    ask(text);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  }

  const styleLabel = (list: StyleOption[] | undefined, id: string | undefined) =>
    list?.find((o) => o.id === id)?.label ?? id ?? "";
  const accentVars = { "--qa-accent": theme.accent, "--qa-tint": theme.tint } as CSSProperties;

  return (
    <TooltipProvider delayDuration={300}>
    <div className="qa-page">
      <h1 className="sr-only">SOP &amp; Healthcare Q&amp;A</h1>
      <div className="qa-shell">
          <aside id="qa-sidebar" className={`qa-sidebar${railMode ? " collapsed" : ""}`} aria-label="Chats">
            {railMode ? (
              <div className="qa-rail">
                <IconTip
                  label="Show chat list"
                  aria-expanded={false}
                  aria-controls="qa-sidebar"
                  onClick={() => setSidebarCollapsed(false)}
                >
                  <PanelLeftOpen />
                </IconTip>
                <IconTip label="New chat" variant="secondary" className="text-primary" onClick={newChat}>
                  <Plus />
                </IconTip>
                <IconTip
                  label="Search chat history"
                  onClick={() => {
                    setSidebarCollapsed(false);
                    setSearchOpen(true);
                  }}
                >
                  <Search />
                </IconTip>
                <Separator className="my-1.5 w-7" />
                {/* Today's chats as persona-coloured dots; the title shows in a tooltip */}
                {todaySessions.slice(0, 8).map((s) => (
                  <IconTip
                    key={s.id}
                    label={s.title}
                    aria-label={`Open chat: ${s.title}`}
                    className={cn("h-8", s.id === activeId && "border border-primary bg-secondary")}
                    onClick={() => openSession(s.id)}
                  >
                    <span className="size-2.5 rounded-full" style={{ background: themeFor(s.persona).accent }} />
                  </IconTip>
                ))}
              </div>
            ) : (
            <>
            <div className="qa-sidebar-head">
              <IconTip
                label="Hide chat list"
                aria-expanded={true}
                aria-controls="qa-sidebar"
                className="-ml-1.5 size-8"
                onClick={() => setSidebarCollapsed(true)}
              >
                <PanelLeftClose />
              </IconTip>
              <span className="qa-sidebar-title">Chats</span>
              <Button size="sm" variant="secondary" className="ml-auto font-bold text-primary" onClick={newChat}>
                <Plus /> New chat
              </Button>
            </div>

            <div className="qa-group-label">Today</div>
            <div className="qa-session-list">
              {todaySessions.length === 0 && (
                <div className="qa-session-empty">No conversations yet today</div>
              )}
              {todaySessions.map((s) =>
                renderSessionRow(
                  s,
                  new Date(s.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                )
              )}
            </div>

            <Button
              variant="ghost"
              size="sm"
              className="mt-2.5 justify-start px-2 font-semibold text-muted-foreground"
              aria-expanded={searchOpen}
              onClick={() => setSearchOpen((v) => !v)}
            >
              <Search /> Search chat history{olderCount > 0 ? ` (${olderCount} earlier)` : ""}
            </Button>

            {sessions.length > 0 && (
              <Button
                variant={pendingDelete === "all" ? "destructive" : "ghost"}
                size="sm"
                className={cn("justify-start px-2 font-semibold", pendingDelete !== "all" && "text-muted-foreground hover:text-destructive")}
                disabled={loading}
                onClick={() => (pendingDelete === "all" ? clearAllSessions() : setPendingDelete("all"))}
                onBlur={() => pendingDelete === "all" && setPendingDelete(null)}
              >
                <Trash2 />
                {pendingDelete === "all"
                  ? `Click again to delete all ${sessions.length} conversation${sessions.length > 1 ? "s" : ""}`
                  : "Clear all history"}
              </Button>
            )}

            {searchOpen && (
              <div className="qa-search-box">
                <input
                  autoFocus
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search all past conversations…"
                />
                <div className="qa-search-results">
                  {searchQuery.trim() === "" && (
                    <div className="qa-search-empty">Type to search every past conversation</div>
                  )}
                  {searchQuery.trim() !== "" && searchResults.length === 0 && (
                    <div className="qa-search-empty">No conversations match "{searchQuery}"</div>
                  )}
                  {searchResults.map((s) =>
                    renderSessionRow(
                      s,
                      new Date(s.updatedAt).toLocaleDateString([], { month: "short", day: "numeric" })
                    )
                  )}
                </div>
              </div>
            )}
            </>
            )}
          </aside>

          <div className={`qa-chat-col${inSetup ? " setup" : ""}`} style={accentVars}>
            {!inSetup && catalogue && effective && activeSession && (
              <div className="qa-chat-head">
                <StyleChip
                  key={activeSession.id}
                  catalogue={catalogue}
                  personaId={effective.persona}
                  length={effective.length}
                  technicalLevel={effective.technical_level}
                  onChange={(change) =>
                    updateSession(activeSession.id, (s) => ({
                      ...s,
                      persona: s.persona ?? effective.persona,
                      length: change.length ?? effective.length,
                      technical_level: change.technical_level ?? effective.technical_level,
                    }))
                  }
                  onNewChat={newChat}
                  disabled={loading}
                />
                <TopicsMenu
                  topics={topics}
                  onAsk={ask}
                  disabled={loading}
                  note={<span className="qa-chat-head-note">Answers from TTSH Eye Centre SOP documents</span>}
                />
              </div>
            )}

            <div ref={threadRef} onScroll={onThreadScroll} className={`qa-thread${inSetup ? " setup" : ""}`}>
              {inSetup && catalogue && prefs && (
                <SetupHero
                  catalogue={catalogue}
                  prefs={prefs}
                  onChange={updatePrefs}
                  prompts={prompts}
                  onAsk={ask}
                  disabled={loading}
                />
              )}

              {inSetup && !(catalogue && prefs) && (
                <div className="qa-empty">
                  <h3>Ask about clinic SOPs, protocols, or healthcare procedures</h3>
                  <div className="muted">Try one of these, or type your own question below</div>
                  <div className="qa-suggestions">
                    {prompts.map((q) => (
                      <button key={q} className="qa-suggestion-btn" onClick={() => ask(q)}>
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={m.id} className={cn("qa-msg user", freshIds.current.has(m.id) && "enter")}>
                    {editingId === m.id ? (
                      <form
                        className="qa-edit"
                        onSubmit={(e) => {
                          e.preventDefault();
                          submitEdit(m.id);
                        }}
                      >
                        <label htmlFor={`qa-edit-${m.id}`} className="sr-only">
                          Edit your question
                        </label>
                        <textarea
                          id={`qa-edit-${m.id}`}
                          autoFocus
                          value={editText}
                          maxLength={1000}
                          rows={Math.min(8, Math.max(2, editText.split("\n").length))}
                          onChange={(e) => setEditText(e.target.value)}
                          onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
                          onKeyDown={(e) => {
                            if (e.key === "Escape") setEditingId(null);
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              submitEdit(m.id);
                            }
                          }}
                        />
                        <div className="qa-edit-actions">
                          <Button type="button" variant="ghost" size="sm" className="rounded-full" onClick={() => setEditingId(null)}>
                            Cancel
                          </Button>
                          <Button type="submit" size="sm" className="qa-send rounded-full px-4" disabled={!editText.trim() || loading}>
                            Send
                          </Button>
                        </div>
                      </form>
                    ) : (
                      <div className="qa-user-col">
                        <div className="qa-bubble">{m.text}</div>
                        <div className="qa-user-actions">
                          <CopyButton text={m.text} label="Copy question" />
                          <EditButton onClick={() => startEdit(m)} disabled={loading} />
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    key={m.id}
                    className={cn("qa-msg assistant", m.isError && "error", freshIds.current.has(m.id) && "enter")}
                  >
                    <Orb theme={theme} size={30} />
                    <div className="qa-msg-body">
                      <div className="qa-msg-meta">
                        <span className="qa-role">{m.personaName ?? effectivePersona?.name ?? "SOP Assistant"}</span>
                        <span className="qa-time">{m.time}</span>
                      </div>
                      {m.thinking && m.thinking.length > 0 && (
                        <details className="qa-thinking">
                          <summary>
                            Thinking · {m.thinking.length} step{m.thinking.length > 1 ? "s" : ""}
                          </summary>
                          <div className="qa-thinking-list">
                            {m.thinking.map((t, j) => (
                              <div key={j} className={`qa-thinking-step ${t.kind}`}>
                                <span className="qa-thinking-kind">{t.kind === "search" ? "Search" : "Reasoning"}</span>
                                {t.text}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                      <div className="qa-answer">
                        {m.isError ? <div className="qa-text">{m.text}</div> : <Markdown text={m.text} />}
                        {m.placeholder && (
                          <div className="qa-chip">placeholder response — KB not connected</div>
                        )}
                        {/* Not covered: offer what the knowledge base does cover (latest reply only) */}
                        {m.covered === false && !m.isError && i === messages.length - 1 && (
                          <TopicHelp topics={topics} onAsk={ask} disabled={loading} />
                        )}
                        {m.sources && m.sources.length > 0 && (
                          <details className="qa-sources">
                            <summary>
                              {m.sources.length} source{m.sources.length > 1 ? "s" : ""}
                            </summary>
                            <div className="qa-sources-list">
                              {m.sources.map((s, j) => (
                                <div key={j} className="qa-source">
                                  <strong>{s.title}</strong> — {s.snippet}
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                      {/* Answer actions — hidden while this answer is still streaming */}
                      {!m.isError && !(loading && i === messages.length - 1) && (m.text || m.stopped) && (
                        <div className="qa-actions">
                          {m.text && <CopyButton text={m.text} />}
                          {m.text && (
                            <AnswerFeedback
                              rating={m.feedback}
                              reason={m.feedbackReason}
                              onRate={(rating, reason) => rateAnswer(m, rating, reason)}
                            />
                          )}
                          {/* Redo only the latest answer: older ones would discard the conversation after them */}
                          {i === messages.length - 1 && (
                            <Tooltip>
                              <TooltipTrigger
                                type="button"
                                aria-label="Regenerate answer"
                                onClick={regenerate}
                                disabled={loading}
                                className={ACTION_CLASS}
                              >
                                <RefreshCw aria-hidden="true" />
                              </TooltipTrigger>
                              <TooltipContent side="bottom">Redo</TooltipContent>
                            </Tooltip>
                          )}
                          {m.stopped && <span className="qa-stopped">Response stopped</span>}
                        </div>
                      )}
                      {/* Follow-ups only on the latest reply — older ones are stale */}
                      {m.followups && m.followups.length > 0 && i === messages.length - 1 && (
                        <div className="qa-followups">
                          <span className="qa-field-label">Ask next</span>
                          <div className="flex flex-wrap gap-2">
                            {m.followups.map((q) => (
                              <Button
                                key={q}
                                variant="outline"
                                disabled={loading}
                                className="h-auto whitespace-normal rounded-full px-4 py-2 text-left text-[12.5px] font-semibold hover:border-(--qa-accent) hover:bg-(--qa-tint)"
                                onClick={() => ask(q)}
                              >
                                <CornerDownRight className="text-muted-foreground" aria-hidden="true" />
                                {q}
                              </Button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )
              )}

              {loading && !streaming && (
                <div className="qa-msg assistant enter">
                  <Orb theme={theme} size={30} pulse />
                  <div className="qa-msg-body">
                    <div className="qa-msg-meta">
                      <span className="qa-role">{effectivePersona?.name ?? "SOP Assistant"}</span>
                    </div>
                    <div className="qa-typing">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {showJump && (
              <div className="qa-jump">
                <Button
                  variant="outline"
                  size="icon"
                  aria-label="Scroll to latest message"
                  className="rounded-full bg-card shadow-sm"
                  onClick={() => scrollToBottom(true)}
                >
                  <ArrowDown />
                </Button>
              </div>
            )}

            <form
              className="qa-composer"
              onSubmit={(e) => {
                e.preventDefault();
                ask(input);
              }}
            >
              <label htmlFor="qa-composer-input" className="sr-only">
                Your question
              </label>
              <textarea
                id="qa-composer-input"
                aria-describedby="qa-composer-hint"
                ref={textareaRef}
                className="qa-textarea"
                rows={inSetup ? 2 : 1}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  autoResize();
                }}
                onKeyDown={handleKeyDown}
                placeholder={
                  inSetup
                    ? `Ask the ${effectivePersona?.name ?? "SOP assistant"} anything about clinic SOPs…`
                    : "Ask a follow-up…"
                }
                maxLength={1000}
              />
              <div className="qa-composer-side">
                {inSetup && effectivePersona && effective && (
                  <span className="qa-composer-summary">
                    {effectivePersona.name} · {styleLabel(catalogue?.lengths, effective.length)} ·{" "}
                    {styleLabel(catalogue?.technical_levels, effective.technical_level)}
                  </span>
                )}
                {loading ? (
                  <Button
                    type="button"
                    size="icon"
                    aria-label="Stop generating"
                    onClick={stopAnswer}
                    className="qa-send size-9 rounded-full"
                  >
                    <Square fill="currentColor" strokeWidth={0} aria-hidden="true" className="size-3.5!" />
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    size="icon"
                    aria-label="Send question"
                    disabled={!input.trim()}
                    className="qa-send size-9 rounded-full [&_svg]:size-[18px]"
                  >
                    <ArrowUp strokeWidth={2.4} aria-hidden="true" />
                  </Button>
                )}
              </div>
            </form>
            {/* Shown while typing (see .qa-composer:focus-within + .qa-composer-hint) */}
            <div className="qa-composer-hint" id="qa-composer-hint">
              <kbd>Enter</kbd> to send · <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line
            </div>
          </div>
      </div>
    </div>
    </TooltipProvider>
  );
}
