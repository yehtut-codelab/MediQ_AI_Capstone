import "./globals.css";
import type { ReactNode } from "react";
import { Encode_Sans } from "next/font/google";
import NavLinks from "@/components/NavLinks";

// Encode Sans is the typeface of nhghealth.com.sg, used here for both text and
// display numbers. Self-hosted and preloaded by Next.js (no render-blocking
// request to Google at page load, no layout shift). Exposed as a CSS variable
// used in legacy.css.
const encodeSans = Encode_Sans({ subsets: ["latin"], variable: "--font-encode-sans", display: "swap" });

export const metadata = {
  title: "MediQ AI — Queue & Wait Advisor",
  description: "RAG-based wait time estimates from historical visit events",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={encodeSans.variable}>
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
