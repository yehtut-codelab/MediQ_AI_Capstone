// Compact summary of a running chat's persona + answer style. Opens a popover
// (shadcn/ui) to change length / technical level mid-chat; the persona stays
// fixed for the conversation (switching means a new chat).

import { useState, type CSSProperties } from "react";
import { ChevronDown, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { themeFor } from "./personaTheme";
import type { Catalogue } from "./types";
import { Orb, Segmented } from "./ui";

type Props = {
  catalogue: Catalogue;
  personaId: string;
  length: string;
  technicalLevel: string;
  onChange: (change: { length?: string; technical_level?: string }) => void;
  onNewChat: () => void;
  disabled: boolean;
};

export default function StyleChip({ catalogue, personaId, length, technicalLevel, onChange, onNewChat, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const persona = catalogue.personas.find((p) => p.id === personaId);
  const theme = themeFor(personaId);
  const label = (list: { id: string; label: string }[], id: string) => list.find((o) => o.id === id)?.label ?? id;
  // The popover renders in a portal, outside the chat column's CSS variables.
  const accentVars = { "--qa-accent": theme.accent, "--qa-tint": theme.tint } as CSSProperties;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger className="qa-style-chip" style={accentVars}>
        <Orb theme={theme} size={22} />
        <span className="qa-style-chip-name">{persona?.name ?? "SOP Assistant"}</span>
        <span className="qa-style-chip-style">
          · {label(catalogue.lengths, length)} · {label(catalogue.technical_levels, technicalLevel)}
        </span>
        <ChevronDown size={14} strokeWidth={2.2} aria-hidden="true" className={open ? "rotate-180 transition-transform" : "transition-transform"} />
      </PopoverTrigger>

      <PopoverContent
        align="start"
        sideOffset={8}
        aria-label="Answer settings"
        className="flex w-[360px] flex-col gap-3.5 rounded-xl p-4 shadow-lg"
        style={accentVars}
      >
        <div className="flex items-start gap-2.5 rounded-lg bg-[#f6f5f1] p-3 text-muted-foreground">
          <Lock size={16} strokeWidth={1.8} className="mt-0.5 shrink-0" aria-hidden="true" />
          <div className="flex flex-col gap-0.5 text-xs leading-relaxed">
            <strong className="text-[12.5px] text-foreground">Persona is fixed for this chat</strong>
            <span>
              {persona?.name ?? "Default persona"}. To switch persona,{" "}
              <Button
                variant="link"
                className="h-auto p-0 text-xs"
                onClick={() => {
                  setOpen(false);
                  onNewChat();
                }}
              >
                start a new chat
              </Button>
              .
            </span>
          </div>
        </div>
        <Segmented
          label="Answer length"
          options={catalogue.lengths}
          value={length}
          onChange={(id) => onChange({ length: id })}
          accent={theme.accent}
          disabled={disabled}
          stretch
        />
        <Segmented
          label="Technical level"
          options={catalogue.technical_levels}
          value={technicalLevel}
          onChange={(id) => onChange({ technical_level: id })}
          accent={theme.accent}
          disabled={disabled}
          stretch
        />
        <span className="text-[11px] text-muted-foreground">Changes apply from your next question.</span>
      </PopoverContent>
    </Popover>
  );
}
