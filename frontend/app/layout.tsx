import "./globals.css";
import type { Metadata } from "next";
import NavBar from "./NavBar";

export const metadata: Metadata = {
  title: "MindGuard",
  description: "AI-powered mental-health text classification and empathetic response demo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <NavBar />
        <div className="container">{children}</div>
      </body>
    </html>
  );
}
