import "./globals.css";
import type { Metadata } from "next";
import { Poppins, Inter } from "next/font/google";
import NavBar from "./NavBar";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-heading",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MindGuard",
  description: "AI-powered mental-health text classification and empathetic response demo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${poppins.variable} ${inter.variable}`}>
      <body>
        <div className="bg-photo" aria-hidden="true" />
        <div className="bg-decor" aria-hidden="true">
          <span className="blob blob-a" />
          <span className="blob blob-b" />
          <span className="blob blob-c" />
        </div>
        <NavBar />
        <div className="container">{children}</div>
      </body>
    </html>
  );
}
