"use client";

// Top navigation links; marks the current page with aria-current so it is both
// highlighted (see .nav-links a[aria-current] in legacy.css) and announced.

import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Arrival Simulator" },
  { href: "/register", label: "New Patient Entry" },
  { href: "/stations", label: "Station Map" },
  { href: "/agent", label: "AI Agent Console" },
  { href: "/qa", label: "SOP Q&A" },
  { href: "/documents", label: "Knowledge Base" },
];

export default function NavLinks() {
  const pathname = usePathname();
  return (
    <div className="nav-links">
      {LINKS.map((l) => {
        const current = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
        return (
          <a key={l.href} href={l.href} aria-current={current ? "page" : undefined}>
            {l.label}
          </a>
        );
      })}
    </div>
  );
}
