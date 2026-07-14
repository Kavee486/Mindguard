"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Avatar from "./Avatar";

const links = [
  { href: "/", label: "Analyze" },
  { href: "/compare", label: "Compare Models" },
  { href: "/interpret", label: "Interpretability" },
];

export default function NavBar() {
  const pathname = usePathname();
  return (
    <nav className="nav">
      <span className="brand">
        <Avatar size={30} />
        MindGuard
      </span>
      {links.map((l) => (
        <Link key={l.href} href={l.href} className={pathname === l.href ? "active" : ""}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
