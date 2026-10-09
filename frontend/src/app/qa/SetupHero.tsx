// Setup screen shown while a chat is empty: greeting orb, persona cards,
// answer-style toggles, a live preview and persona-specific suggested questions.
// Once the first question is sent it gives way to the conversation, and the
// choices collapse into the StyleChip.

import type { CSSProperties } from "react";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { buildPreview, PREVIEW_QUESTION, themeFor } from "./personaTheme";
import type { Catalogue, Prefs } from "./types";
import { Orb, PersonaIcon, Segmented } from "./ui";

type Props = {
  catalogue: Catalogue;
  prefs: Prefs;
  onChange: (change: Partial<Prefs>) => void;
  prompts: string[];
  onAsk: (question: string) => void;
  disabled: boolean;
};

export default function SetupHero({ catalogue, prefs, onChange, prompts, onAsk, disabled }: Props) {
  const persona = catalogue.personas.find((p) => p.id === prefs.persona) ?? catalogue.personas[0];
  const theme = themeFor(persona.id);
  const label = (list: { id: string; label: string }[], id: string) => list.find((o) => o.id === id)?.label ?? id;
  const customised =
    prefs.length !== persona.default_length || prefs.technical_level !== persona.default_technical_level;
  const preview = buildPreview(persona.id, prefs.length, prefs.technical_level);
  const words = preview?.reduce((n, l) => n + l.text.split(/\s+/).length, 0) ?? 0;
  const accentVars = { "--qa-accent": theme.accent, "--qa-tint": theme.tint } as CSSProperties;

  return (
    <div className="qa-hero" style={accentVars}>
      <div className="qa-hero-greeting">
        <Orb theme={theme} size={48} />
        <div>
          <h2 className="qa-hero-title">Hi, I&apos;m MediQ. Who am I helping today?</h2>
          <p className="qa-hero-sub">
            Pick a persona and answer style — or just start typing below. Answers come only from TTSH Eye
            Centre SOP documents.
          </p>
        </div>
      </div>

      <div role="group" aria-label="Persona" className="qa-persona-grid">
        {catalogue.personas.map((p) => {
          const t = themeFor(p.id);
          const on = p.id === persona.id;
          return (
            <button
              key={p.id}
              type="button"
              className={`qa-persona-card${on ? " on" : ""}`}
              aria-pressed={on}
              disabled={disabled}
              style={{ "--card-accent": t.accent, "--card-tint": t.tint } as CSSProperties}
              onClick={() => onChange({ persona: p.id })}
            >
              <span className="qa-persona-icon">
                <PersonaIcon icon={t.icon} />
              </span>
              <span className="qa-persona-name">{p.name}</span>
              <span className="qa-persona-audience">{p.audience}</span>
              <span className="qa-persona-tag">
                {label(catalogue.lengths, p.default_length)} · {label(catalogue.technical_levels, p.default_technical_level)}
              </span>
            </button>
          );
        })}
      </div>

      <div className="qa-hero-controls">
        <Segmented
          label="Answer length"
          options={catalogue.lengths}
          value={prefs.length}
          onChange={(id) => onChange({ length: id })}
          accent={theme.accent}
          disabled={disabled}
        />
        <Segmented
          label="Technical level"
          options={catalogue.technical_levels}
          value={prefs.technical_level}
          onChange={(id) => onChange({ technical_level: id })}
          accent={theme.accent}
          disabled={disabled}
        />
        {customised && (
          <Button
            variant="link"
            className="h-9 px-1 text-[12.5px] font-semibold"
            disabled={disabled}
            onClick={() =>
              onChange({ length: persona.default_length, technical_level: persona.default_technical_level })
            }
          >
            <RotateCcw aria-hidden="true" />
            Reset to persona default
          </Button>
        )}
      </div>

      {preview && (
        <section className="qa-preview" aria-label="Answer preview" aria-live="polite">
          <div className="qa-preview-head">
            <span className="qa-field-label">Live preview · how answers will read</span>
            <span className="qa-preview-meta">{words} words</span>
          </div>
          <div className="qa-preview-q">“{PREVIEW_QUESTION}”</div>
          <div className="qa-preview-body">
            <Orb theme={theme} size={24} />
            <div className="qa-preview-text">
              <span className="qa-preview-name">{persona.name}</span>
              {preview.map((line, i) =>
                line.kind === "head" ? (
                  <div key={i} className="qa-preview-h">{line.text}</div>
                ) : line.kind === "bullet" ? (
                  <div key={i} className="qa-preview-li">{line.text}</div>
                ) : (
                  <div key={i}>{line.text}</div>
                )
              )}
            </div>
          </div>
        </section>
      )}

      <div className="qa-hero-prompts">
        <span className="qa-field-label">Try asking</span>
        <div className="flex flex-wrap gap-2">
          {prompts.map((q) => (
            <Button
              key={q}
              variant="outline"
              disabled={disabled}
              className="h-auto whitespace-normal rounded-full px-4 py-2 text-left text-[12.5px] font-semibold hover:border-(--qa-accent) hover:bg-(--qa-tint)"
              onClick={() => onAsk(q)}
            >
              {q}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
