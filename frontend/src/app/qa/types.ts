// Shared types for the SOP Q&A page.

// Persona + answer-style catalogue served by GET /api/v1/qa/personas
export type PersonaOption = {
  id: string;
  name: string;
  audience: string;
  default_length: string;
  default_technical_level: string;
};
export type StyleOption = { id: string; label: string };
export type Catalogue = {
  default_persona: string;
  personas: PersonaOption[];
  lengths: StyleOption[];
  technical_levels: StyleOption[];
};

/** Persona + answer style: the setup screen's choice, and each session's snapshot. */
export type Prefs = { persona: string; length: string; technical_level: string };

/** A knowledge-base topic from GET /api/v1/qa/topics (identical uploads grouped). */
export type Topic = {
  id: string;
  name: string;
  kind: string;
  summary: string;
  example_questions: string[];
  source_name: string;
  chunk_count: number;
  documents: { id: string; name: string; uploaded_at: string }[];
  duplicate_count: number;
  updated_at: string;
};

export type Source = { title: string; snippet: string; score?: number };
export type ThinkingStep = { kind: "reasoning" | "search"; text: string };
export type Message = {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  thinking?: ThinkingStep[];
  followups?: string[];
  personaName?: string;
  /** false when the knowledge base doesn't cover the question (UI offers topics). */
  covered?: boolean;
  /** The user pressed Stop while this answer was streaming. */
  stopped?: boolean;
  /** Thumbs up/down on this answer (also sent to POST /qa/feedback). */
  feedback?: "up" | "down";
  feedbackReason?: string;
  placeholder?: boolean;
  isError?: boolean;
  time: string;
};
export type Session = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  // Persona is fixed when the chat starts; length/technical level can change
  // mid-chat. Absent on chats saved before personas existed.
  persona?: string;
  length?: string;
  technical_level?: string;
};
