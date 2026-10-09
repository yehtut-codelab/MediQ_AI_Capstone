// Minimal Markdown for chat answers: headings, **bold**, *italic*, `code`,
// ordered/unordered lists and paragraphs. Builds React elements directly —
// never innerHTML — so model output cannot inject markup. Tolerates partial
// text while an answer is still streaming.

import { Fragment, type ReactNode } from "react";

type Block =
  | { kind: "heading"; text: string }
  | { kind: "ol" | "ul"; items: string[] }
  | { kind: "p"; lines: string[] };

const HEADING = /^\s{0,3}#{1,6}\s+(.*)$/;
const ORDERED = /^\s*\d+[.)]\s+(.*)$/;
const BULLET = /^\s*[-*•]\s+(.*)$/;

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  const flushPara = () => {
    if (para.length) blocks.push({ kind: "p", lines: para });
    para = [];
  };

  for (const line of text.split("\n")) {
    const heading = line.match(HEADING);
    const ordered = line.match(ORDERED);
    const bullet = line.match(BULLET);
    if (!line.trim()) {
      flushPara();
    } else if (heading) {
      flushPara();
      blocks.push({ kind: "heading", text: heading[1] });
    } else if (ordered || bullet) {
      flushPara();
      const kind = ordered ? "ol" : "ul";
      const item = (ordered ?? bullet)![1];
      const last = blocks[blocks.length - 1];
      if (last && last.kind === kind) last.items.push(item);
      else blocks.push({ kind, items: [item] });
    } else {
      para.push(line.trim());
    }
  }
  flushPara();
  return blocks;
}

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;

function inline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if ((part.startsWith("*") && part.endsWith("*") || part.startsWith("_") && part.endsWith("_")) && part.length > 2) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export default function Markdown({ text }: { text: string }) {
  return (
    <div className="qa-md">
      {parseBlocks(text).map((b, i) => {
        if (b.kind === "heading") return <h4 key={i}>{inline(b.text)}</h4>;
        if (b.kind === "p") {
          return (
            <p key={i}>
              {b.lines.map((line, j) => (
                <Fragment key={j}>
                  {j > 0 && <br />}
                  {inline(line)}
                </Fragment>
              ))}
            </p>
          );
        }
        const List = b.kind;
        return (
          <List key={i}>
            {b.items.map((item, j) => (
              <li key={j}>{inline(item)}</li>
            ))}
          </List>
        );
      })}
    </div>
  );
}
