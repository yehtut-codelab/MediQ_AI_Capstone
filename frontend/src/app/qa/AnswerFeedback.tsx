// Thumbs up / thumbs down on an answer. A thumbs down also asks (optionally)
// what went wrong. The parent stores the rating and sends it to the backend.

import { useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { Popover, PopoverAnchor, PopoverContent } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const REASONS = ["Inaccurate", "Not in the SOPs", "Not helpful", "Too long", "Too technical", "Wrong tone"];

const ICON_BTN = cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "size-8 text-muted-foreground");

type Props = {
  rating?: "up" | "down";
  reason?: string;
  /** undefined rating clears it; reason only applies to "down". */
  onRate: (rating: "up" | "down" | undefined, reason?: string) => void;
};

export default function AnswerFeedback({ rating, reason, onRate }: Props) {
  const [askWhy, setAskWhy] = useState(false);
  const [thanks, setThanks] = useState(false);

  function thumbsDown() {
    if (rating === "down") {
      onRate(undefined);
      setAskWhy(false);
    } else {
      onRate("down");
      setThanks(false);
      setAskWhy(true);
    }
  }

  function pickReason(r: string) {
    onRate("down", r);
    setThanks(true);
    setTimeout(() => setAskWhy(false), 900);
  }

  return (
    <>
      <Tooltip>
        <TooltipTrigger
          type="button"
          aria-label={rating === "up" ? "Remove good response rating" : "Good response"}
          aria-pressed={rating === "up"}
          onClick={() => onRate(rating === "up" ? undefined : "up")}
          className={cn(ICON_BTN, rating === "up" && "text-foreground")}
        >
          <ThumbsUp fill={rating === "up" ? "currentColor" : "none"} aria-hidden="true" />
        </TooltipTrigger>
        <TooltipContent side="bottom">Good response</TooltipContent>
      </Tooltip>

      <Popover open={askWhy} onOpenChange={setAskWhy}>
        <PopoverAnchor asChild>
          <span className="inline-flex">
            <Tooltip>
              <TooltipTrigger
                type="button"
                aria-label={rating === "down" ? "Remove bad response rating" : "Bad response"}
                aria-pressed={rating === "down"}
                onClick={thumbsDown}
                className={cn(ICON_BTN, rating === "down" && "text-foreground")}
              >
                <ThumbsDown fill={rating === "down" ? "currentColor" : "none"} aria-hidden="true" />
              </TooltipTrigger>
              <TooltipContent side="bottom">Bad response</TooltipContent>
            </Tooltip>
          </span>
        </PopoverAnchor>
        <PopoverContent
          align="start"
          sideOffset={6}
          aria-label="What went wrong?"
          className="flex w-[300px] flex-col gap-2.5 rounded-xl p-3.5 shadow-lg"
        >
          {thanks ? (
            <span className="py-1 text-[13px] font-semibold">Thanks for the feedback.</span>
          ) : (
            <>
              <span className="text-[13px] font-semibold">What went wrong? (optional)</span>
              <div className="flex flex-wrap gap-1.5">
                {REASONS.map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => pickReason(r)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-[12px] font-semibold transition-colors hover:bg-[#f5f5f2]",
                      reason === r && "border-foreground bg-[#f0f0ed]"
                    )}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </>
          )}
        </PopoverContent>
      </Popover>
    </>
  );
}
