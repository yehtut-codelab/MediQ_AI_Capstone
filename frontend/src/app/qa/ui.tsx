// Small visual pieces shared by the SOP Q&A page: the persona orb and icons,
// plus the answer-style toggle (shadcn/ui ToggleGroup).

import type { CSSProperties } from "react";
import { ClipboardList, Eye, GraduationCap, HeartHandshake, Headset, type LucideIcon } from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import type { PersonaTheme } from "./personaTheme";

/** The glowing "AI presence" orb, tinted per persona. `pulse` adds the halo ring. */
export function Orb({ theme, size, pulse = false }: { theme: PersonaTheme; size: number; pulse?: boolean }) {
  const vars = { "--orb-accent": theme.accent, "--orb-tint": theme.tint, width: size, height: size } as CSSProperties;
  return (
    <span className={`qa-orb${pulse ? " pulse" : ""}`} style={vars} aria-hidden="true">
      <span className="qa-orb-core" />
    </span>
  );
}

const ICONS: Record<PersonaTheme["icon"], LucideIcon> = {
  clipboard: ClipboardList,
  eye: Eye,
  heart: HeartHandshake,
  headset: Headset,
  cap: GraduationCap,
};

export function PersonaIcon({ icon, size = 20 }: { icon: PersonaTheme["icon"]; size?: number }) {
  const Icon = ICONS[icon];
  return <Icon size={size} strokeWidth={1.8} aria-hidden="true" />;
}

/** Three-way toggle (Concise / Balanced / Comprehensive etc.), tinted with the persona accent. */
export function Segmented({
  label,
  options,
  value,
  onChange,
  accent,
  disabled,
  stretch = false,
}: {
  label: string;
  options: { id: string; label: string }[];
  value: string;
  onChange: (id: string) => void;
  accent: string;
  disabled?: boolean;
  stretch?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="qa-field-label">{label}</span>
      <ToggleGroup
        type="single"
        aria-label={label}
        value={value}
        // Radix sends "" when the active item is clicked again; keep one selected.
        onValueChange={(v) => v && onChange(v)}
        disabled={disabled}
        spacing={1}
        className={cn("rounded-[10px] bg-[#efeee9] p-[3px]", stretch && "flex w-full")}
        style={{ "--seg-accent": accent } as CSSProperties}
      >
        {options.map((o) => (
          <ToggleGroupItem
            key={o.id}
            value={o.id}
            className={cn(
              "h-9 rounded-lg px-3.5 text-[12.5px] font-bold text-muted-foreground hover:bg-transparent hover:text-foreground",
              "data-[state=on]:bg-card data-[state=on]:text-(--seg-accent) data-[state=on]:shadow-[0_1px_3px_rgba(11,47,82,0.14)]",
              stretch && "flex-1"
            )}
          >
            {o.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}
