import "./globals.css";
import type { ReactNode } from "react";
import { Fraunces, Manrope } from "next/font/google";
import NavLinks from "@/components/NavLinks";

// Self-hosted and preloaded by Next.js (no render-blocking request to Google
// at page load, no layout shift). Exposed as CSS variables used in legacy.css.
const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope", display: "swap" });
const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", axes: ["opsz"], display: "swap" });

export const metadata = {
  title: "MediQ AI — Queue & Wait Advisor",
  description: "RAG-based wait time estimates from historical visit events",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${manrope.variable} ${fraunces.variable}`}>
      <body>
        <nav className="mediq-nav">
          <div>
            <div className="logo-name">MediQ AI</div>
            <div className="logo-sub">AI-Driven Digital Twin for Eye Clinic</div>
          </div>
          <NavLinks />
        </nav>
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
