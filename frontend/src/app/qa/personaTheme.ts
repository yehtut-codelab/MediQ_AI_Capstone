// Presentation data for the SOP Q&A personas. The persona list itself comes
// from the backend (GET /api/v1/qa/personas); this file only adds the look,
// the setup-screen preview and suggested questions per persona id. A persona
// the backend adds without an entry here still works: default colours, no
// preview, the general question pool.

export type PersonaTheme = {
  accent: string; // text-safe on white and on `tint`
  tint: string;
  icon: "clipboard" | "eye" | "heart" | "headset" | "cap";
};

const DEFAULT_THEME: PersonaTheme = { accent: "#1668c9", tint: "#eaf2fd", icon: "clipboard" };

const THEMES: Record<string, PersonaTheme> = {
  clinical_staff: { accent: "#1668c9", tint: "#eaf2fd", icon: "clipboard" },
  clinician: { accent: "#0b2f52", tint: "#e7edf4", icon: "eye" },
  patient: { accent: "#1a6b3c", tint: "#e8f5ee", icon: "heart" },
  front_desk: { accent: "#8a6307", tint: "#fdf6e3", icon: "headset" },
  trainee: { accent: "#6b3fa0", tint: "#f2ecf9", icon: "cap" },
};

export function themeFor(personaId: string | undefined): PersonaTheme {
  return (personaId && THEMES[personaId]) || DEFAULT_THEME;
}

// Suggested questions shown on the setup screen, sampled per new chat.
// Keep every one answerable from the ingested SOP documents (pupil dilation
// SOP, contact lens care brochure) — each was checked against /qa/ask.
const GENERAL_POOL = [
  "What is the SOP for pupil dilation before OCT imaging?",
  "Which patients should not receive dilating eye drops?",
  "How long should we wait after instilling Tropicamide before OCT?",
  "What should patients be told after their pupils are dilated?",
  "What needs to be documented after giving dilating drops?",
  "How many hours a day can contact lenses be worn?",
  "Can I sleep, swim or shower with contact lenses in?",
  "What are the essential dos and don'ts of contact lens care?",
];

const PROMPT_POOLS: Record<string, string[]> = {
  clinical_staff: [
    "What is the SOP for pupil dilation before OCT imaging?",
    "Which patients should not receive dilating eye drops?",
    "How long should we wait after instilling Tropicamide before OCT?",
    "What needs to be documented after giving dilating drops?",
    "What should patients be told after their pupils are dilated?",
    "What symptoms mean a contact lens wearer should stop and seek help?",
  ],
  clinician: [
    "What dose of Tropicamide is used for pupil dilation?",
    "What are the contraindications to routine pupil dilation?",
    "How long until full mydriasis before OCT?",
    "What must be documented after mydriatic drops?",
    "What are the recommended contact lens wearing limits?",
  ],
  // Patient Explainer: staff asking for patient-friendly wording.
  patient: [
    "How do I explain to a patient why they can't drive after dilation?",
    "How do I tell a patient how long their vision will be blurry after the drops?",
    "How do I explain contact lens do's and don'ts to a new wearer?",
    "What should I tell a patient about how long to wear contact lenses each day?",
    "What should I tell a patient about cleaning lenses with tap water?",
    "How do I explain when a contact lens wearer should stop and get help?",
  ],
  front_desk: [
    "What should I tell patients about driving after dilation?",
    "How long do the dilating drops take before the OCT scan?",
    "What do I tell patients about contact lens wearing time?",
    "What should patients expect after their pupils are dilated?",
    "Can patients swim or shower with contact lenses in?",
  ],
  trainee: [
    "Walk me through the pupil dilation SOP before OCT",
    "What checks are needed before giving dilating drops?",
    "What after-care advice follows pupil dilation?",
    "What are the essential dos and don'ts of contact lens care?",
    "What should be done with the solution in the contact lens case each day?",
  ],
};

export function pickPrompts(personaId: string | undefined, count = 3): string[] {
  const pool = [...((personaId && PROMPT_POOLS[personaId]) || GENERAL_POOL)];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, count);
}

// Setup-screen live preview: a hand-written sample answer to one question,
// assembled per persona / length / technical level so it updates instantly.
export const PREVIEW_QUESTION = "What should patients be told after their pupils are dilated?";

export type PreviewLine = { kind: "head" | "bullet" | "para"; text: string };

type Terms = { drops: string; dropsStaff: string; blur: string; light: string };
const TERMS: Record<string, Terms> = {
  plain: { drops: "eye drops that widen your pupils", dropsStaff: "the eye drops", blur: "blurry", light: "extra sensitive to bright light" },
  standard: { drops: "dilating drops", dropsStaff: "dilating drops", blur: "blurred", light: "sensitive to light" },
  technical: { drops: "Tropicamide 1% (mydriatic)", dropsStaff: "Tropicamide 1%", blur: "blurred", light: "photophobic" },
};

const SENTENCES: Record<string, (t: Terms, technical: boolean) => string[]> = {
  clinical_staff: (t) => [
    `Advise the patient that vision may be ${t.blur} and ${t.light} for 3–4 hours after ${t.dropsStaff}.`,
    "Recommend sunglasses and no driving until the effects have resolved.",
    "Document the drop given, the time and the patient's consent.",
    `Before instilling, confirm no narrow-angle glaucoma history or allergy to ${t.dropsStaff}.`,
  ],
  clinician: (t, technical) => [
    `Post-${technical ? "mydriasis" : "dilation"}: ${t.blur} vision, ${t.light}, 3–4 h.`,
    "Advise sunglasses; no driving until resolved.",
    "Record agent, time and consent.",
    "Contraindications: narrow-angle glaucoma, mydriatic allergy.",
  ],
  patient: (t) => [
    `You can tell the patient: “After the ${t.drops}, your vision may be ${t.blur} and your eyes ${t.light} for about 3–4 hours.”`,
    "“Wear sunglasses if it helps, and please don't drive until your eyesight is back to normal.”",
    "“If you're worried about your own eyes, please check with your eye doctor.”",
    "If the patient reports sudden vision loss or severe eye pain, escalate to the doctor on duty or send them to A&E.",
  ],
  front_desk: (t) => [
    `“Your vision may be ${t.blur} and your eyes ${t.light} for 3–4 hours.”`,
    "“Please wear sunglasses if needed, and don't drive until your vision is back to normal.”",
    "Pass questions about the eye condition itself to the nurse on duty.",
    "If a patient feels unwell after the drops, alert the nurse straight away.",
  ],
  trainee: (t) => [
    `Tell the patient their vision may be ${t.blur} and ${t.light} for 3–4 hours.`,
    "Recommend sunglasses and no driving until it resolves — blurred vision makes driving unsafe.",
    "Document the drop, the time and consent in the clinical record.",
    "Remember: 3–4 hours · sunglasses · no driving.",
  ],
};

/** null when the persona has no hand-written preview (e.g. one added on the backend only). */
export function buildPreview(personaId: string, length: string, technical: string): PreviewLine[] | null {
  const make = SENTENCES[personaId];
  if (!make) return null;
  const sentences = make(TERMS[technical] ?? TERMS.standard, technical === "technical");
  if (length === "concise") return [{ kind: "para", text: sentences.slice(0, 2).join(" ") }];
  if (length === "balanced") return sentences.slice(0, 3).map((text) => ({ kind: "bullet", text }));
  return [
    { kind: "head", text: personaId === "patient" ? "Explaining it to the patient" : "What to tell the patient" },
    ...sentences.map((text) => ({ kind: "bullet" as const, text })),
  ];
}
