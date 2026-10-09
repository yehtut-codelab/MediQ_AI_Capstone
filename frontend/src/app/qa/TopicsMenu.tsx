// "Topics" button in the chat header: a popover (shadcn/ui) listing what the
// knowledge base covers, each topic with its summary and clickable example
// questions. Also TopicHelp, shown under a "not covered" answer.

import { useState, type ReactNode } from "react";
import { ArrowRight, BookOpen, CornerDownRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import type { Topic } from "./types";

type Props = {
  topics: Topic[];
  onAsk: (question: string) => void;
  disabled: boolean;
};

export function formatUpdated(topics: Topic[]): string {
  const latest = topics.map((t) => t.updated_at).filter(Boolean).sort().pop();
  return latest ? new Date(latest).toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" }) : "";
}

export function KindBadge({ kind }: { kind: string }) {
  return (
    <Badge variant="secondary" className="rounded-full px-2 text-[10.5px] font-bold text-primary">
      {kind}
    </Badge>
  );
}

/**
 * Sits at the right end of the chat header (after `note`); the popover opens
 * right-aligned under the button, so button and panel line up and the panel
 * drops into the empty right margin rather than over the answer.
 */
export default function TopicsMenu({ topics, onAsk, disabled, note }: Props & { note: ReactNode }) {
  const [open, setOpen] = useState(false);
  if (topics.length === 0) return <span className="ml-auto">{note}</span>;
  const updated = formatUpdated(topics);

  return (
    <div className="ml-auto flex items-center gap-3">
    {note}
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        className={cn(
          "inline-flex h-9 cursor-pointer items-center gap-1.5 rounded-full border bg-card px-3.5 text-[12.5px] font-bold text-foreground transition-colors",
          "hover:border-[#d3d2ca] hover:bg-[#f5f5f2] data-[state=open]:border-[#d3d2ca] data-[state=open]:bg-[#f5f5f2]"
        )}
      >
        <BookOpen size={15} strokeWidth={1.8} aria-hidden="true" />
        Topics · {topics.length}
      </PopoverTrigger>

      <PopoverContent
        align="end"
        sideOffset={8}
        aria-label="What I know"
        className="flex max-h-[min(72vh,640px)] w-[440px] max-w-[calc(100vw-24px)] flex-col gap-1 rounded-2xl p-0 shadow-lg"
      >
        <div className="flex items-baseline justify-between gap-3 px-5 pt-4 pb-2">
          <span className="text-base font-bold">What I know</span>
          <span className="text-[11.5px] text-muted-foreground">
            {topics.length} topic{topics.length > 1 ? "s" : ""}
            {updated && ` · updated ${updated}`}
          </span>
        </div>

        <div className="overflow-y-auto px-2">
          {topics.map((t, i) => (
            <section key={t.id} className={cn("flex flex-col gap-1.5 px-3 py-3", i > 0 && "border-t")}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-bold">{t.name}</span>
                <KindBadge kind={t.kind} />
              </div>
              <p className="text-[12.5px] leading-relaxed text-muted-foreground">{t.summary}</p>
              <div className="-mx-2 mt-0.5 flex flex-col">
                {t.example_questions.slice(0, 3).map((q) => (
                  <Button
                    key={q}
                    variant="ghost"
                    disabled={disabled}
                    className="group h-auto justify-start gap-2 whitespace-normal rounded-lg px-2 py-1.5 text-left text-[12.5px] font-semibold hover:bg-secondary hover:text-secondary-foreground"
                    onClick={() => {
                      setOpen(false);
                      onAsk(q);
                    }}
                  >
                    <CornerDownRight className="text-muted-foreground group-hover:text-primary" aria-hidden="true" />
                    {q}
                  </Button>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="flex items-center justify-between border-t px-5 py-3 text-[11.5px] text-muted-foreground">
          <span>Click a question to ask it.</span>
          <a href="/documents" className="inline-flex items-center gap-1 font-semibold text-primary hover:underline">
            Open Knowledge Base <ArrowRight size={13} aria-hidden="true" />
          </a>
        </div>
      </PopoverContent>
    </Popover>
    </div>
  );
}

/** Shown under a "not covered" answer: what the knowledge base can help with. */
export function TopicHelp({ topics, onAsk, disabled }: Props) {
  const [openId, setOpenId] = useState<string>(topics[0]?.id ?? "");
  if (topics.length === 0) return null;
  const openTopic = topics.find((t) => t.id === openId);

  return (
    <div className="mt-3 flex flex-col gap-2.5 rounded-xl bg-[#f6f5f1] px-4 py-3">
      <div className="flex items-center gap-2 text-[12.5px] font-bold">
        <BookOpen size={16} strokeWidth={1.8} aria-hidden="true" />
        I can currently help with
      </div>
      <ToggleGroup
        type="single"
        aria-label="Knowledge-base topics"
        value={openId}
        onValueChange={setOpenId}
        spacing={2}
        className="flex-wrap"
      >
        {topics.map((t) => (
          <ToggleGroupItem
            key={t.id}
            value={t.id}
            className={cn(
              "h-auto gap-1.5 rounded-[10px] border bg-card px-3 py-1.5 text-[12.5px] font-bold",
              "data-[state=on]:border-(--qa-accent) data-[state=on]:bg-(--qa-tint) data-[state=on]:text-(--qa-accent)"
            )}
          >
            {t.name} <span className="text-[10.5px] font-semibold text-muted-foreground">{t.kind}</span>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      {openTopic && (
        <div className="flex flex-wrap gap-2">
          {openTopic.example_questions.slice(0, 2).map((q) => (
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
      )}
    </div>
  );
}
